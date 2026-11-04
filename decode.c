/*
 * sonic-decode: reads audio for the sidecar through the image's libav* libraries, in one thread,
 * instead of running the ffmpeg CLI, which hands every frame between its threads.
 *
 *   sonic-decode hash FILE
 *     SHA-256 of the first audio stream's packets, as
 *     `ffmpeg -i FILE -map 0:a:0 -c copy -f hash -hash sha256 -` prints it.
 *   sonic-decode windows FILE FIRST STEP SPAN
 *     The SPAN-sample windows at FIRST + k*STEP (k = 0, 1, ...) of the stream
 *     `ffmpeg -threads 1 -i FILE -map 0:a:0 -ac 1 -ar 16000 -f f32le -` writes, as raw f32le,
 *     samples before 0 as zeros, up to the first window the stream does not fill. The bytes
 *     are the same; where that holds exactly, only the packets around the windows are decoded.
 *
 * Exit 1, with one line on stderr, when the file cannot be opened, has no audio stream or a
 * decoder or filter cannot be set up. Packets the decoder rejects are skipped, as by the CLI.
 */

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/crc.h>
#include <libavutil/downmix_info.h>
#include <libavutil/hash.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/mathematics.h>

#define OUT_RATE 16000
/* Output samples resampled ahead of a window. The resampler starts from mirrored history, so its
 * first output samples differ from the full decode's; that edge is shorter than this. */
#define MARGIN 64
/* Packets decoded ahead of a window's first input sample, which refill the decoder's state
 * (MP3's bit reservoir and overlap). */
#define PREROLL 10
/* Frames at a segment's start whose pts must agree with their sample counts. */
#define CHECKED 3
#define RING (PREROLL + 2)

static void fail(const char *what, int err)
{
    if (err)
        fprintf(stderr, "%s: %s\n", what, av_err2str(err));
    else
        fprintf(stderr, "%s\n", what);
    exit(1);
}

/* ------------------------------------------------------------------ input -- */

static AVFormatContext *fmt;
static AVStream *st;

static void open_input(const char *path)
{
    int ret;

    avformat_close_input(&fmt);
    if ((ret = avformat_open_input(&fmt, path, NULL, NULL)) < 0)
        fail("open", ret);
    if ((ret = avformat_find_stream_info(fmt, NULL)) < 0)
        fail("stream info", ret);
    /* The first audio stream, as -map 0:a:0; the others (cover art) are not read. */
    st = NULL;
    for (unsigned i = 0; i < fmt->nb_streams; i++) {
        if (!st && fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
            st = fmt->streams[i];
        else
            fmt->streams[i]->discard = AVDISCARD_ALL;
    }
    if (!st)
        fail("no audio stream", 0);
}

/* The stream's next packet, 0 at the end. A read error ends the stream, as in the CLI. */
static int next_packet(AVPacket *pkt)
{
    int ret;

    while ((ret = av_read_frame(fmt, pkt)) >= 0 || ret == AVERROR(EAGAIN)) {
        if (ret >= 0 && pkt->stream_index == st->index)
            return 1;
        av_packet_unref(pkt);
    }
    return 0;
}

static int hash(const char *path)
{
    struct AVHashContext *h;
    uint8_t hex[2 * AV_HASH_MAX_SIZE + 1];
    AVPacket *pkt = av_packet_alloc();

    open_input(path);
    if (!pkt || av_hash_alloc(&h, "SHA256") < 0)
        fail("out of memory", 0);
    av_hash_init(h);
    while (next_packet(pkt)) {
        av_hash_update(h, pkt->data, pkt->size);
        av_packet_unref(pkt);
    }
    av_hash_final_hex(h, hex, sizeof(hex));
    printf("%s\n", hex);
    return 0;
}

/* ----------------------------------------------------------------- output -- */

static int64_t first, step, k; /* window layout; k is the next window to write */
static int span;
static float *buf, *zeros;
static int64_t base;           /* 16 kHz position of buf[0] */
static int len, cap;

static int64_t window(void)
{
    return first + k * step;
}

/* Takes n output samples at 16 kHz position pos and writes every window they complete. A pos
 * that does not continue the previous call starts a new segment. */
static void put(int64_t pos, const float *s, int n)
{
    int64_t w, drop;

    if (pos != base + len) {
        base = pos;
        len = 0;
    }
    if (len + n > cap) {
        cap = FFMAX(2 * cap, len + n);
        if (!(buf = realloc(buf, cap * sizeof(*buf))))
            fail("out of memory", 0);
    }
    memcpy(buf + len, s, n * sizeof(*buf));
    len += n;
    for (;;) {
        int nz;

        drop = FFMIN(FFMAX(window(), 0) - base, len);
        if (drop > 0) {
            memmove(buf, buf + drop, (len - drop) * sizeof(*buf));
            base += drop;
            len -= drop;
        }
        if ((w = window()) + span > base + len)
            break;
        nz = FFMIN(w < 0 ? -w : 0, span);
        if (base > w + nz)
            fail("window starts before its segment", 0);
        fwrite(zeros, sizeof(*zeros), nz, stdout);
        fwrite(buf + (w + nz - base), sizeof(*buf), span - nz, stdout);
        k++;
    }
}

/* --------------------------------------------------------------- decoding -- */

static AVCodecContext *decoder_open(void)
{
    const AVCodec *codec = avcodec_find_decoder(st->codecpar->codec_id);
    AVCodecContext *dec;
    AVDictionary *opts = NULL;
    int ret;

    if (!codec)
        fail("no decoder", 0);
    if (!(dec = avcodec_alloc_context3(codec)))
        fail("out of memory", 0);
    if ((ret = avcodec_parameters_to_context(dec, st->codecpar)) < 0)
        fail("decoder", ret);
    dec->pkt_timebase = st->time_base;
    av_dict_set(&opts, "threads", "1", 0);
    ret = avcodec_open2(dec, codec, &opts);
    av_dict_free(&opts);
    if (ret < 0)
        fail("decoder", ret);
    return dec;
}

typedef struct {
    AVFilterGraph *graph;
    AVFilterContext *src, *sink;
    int format, rate, downmix;
    AVChannelLayout layout;
    AVDownmixInfo downmix_info;
    int64_t in, out; /* samples pushed in; 16 kHz position of the next output sample */
} Graph;

/* abuffer -> aformat -> abuffersink from the frame's parameters, as the CLI builds it for
 * -ac 1 -ar 16000 -f f32le; libavfilter inserts the same aresample in between. */
static void graph_open(Graph *g, const AVFrame *f, int64_t out)
{
    const AVFrameSideData *dmx = av_frame_get_side_data(f, AV_FRAME_DATA_DOWNMIX_INFO);
    AVBufferSrcParameters *par = av_buffersrc_parameters_alloc();
    AVFilterContext *format;
    char args[512], layout[256];
    int ret;

    if (!par || !(g->graph = avfilter_graph_alloc()))
        fail("out of memory", 0);
    g->graph->nb_threads = 1;
    if (av_channel_layout_check(&f->ch_layout) && f->ch_layout.order != AV_CHANNEL_ORDER_UNSPEC) {
        av_channel_layout_describe(&f->ch_layout, layout, sizeof(layout));
        snprintf(args, sizeof(args), "time_base=1/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%s",
                 f->sample_rate, f->sample_rate, av_get_sample_fmt_name(f->format), layout);
    } else
        snprintf(args, sizeof(args), "time_base=1/%d:sample_rate=%d:sample_fmt=%s:channels=%d",
                 f->sample_rate, f->sample_rate, av_get_sample_fmt_name(f->format),
                 f->ch_layout.nb_channels);
    /* The CLI hands the source the frame's global side data and its downmix info, which
     * aresample reads when it is set up. */
    for (int i = 0; i < f->nb_side_data; i++) {
        const AVFrameSideData *sd = f->side_data[i];
        const AVSideDataDescriptor *desc = av_frame_side_data_desc(sd->type);

        if ((desc && desc->props & AV_SIDE_DATA_PROP_GLOBAL && sd->type != AV_FRAME_DATA_DISPLAYMATRIX) ||
            sd->type == AV_FRAME_DATA_DOWNMIX_INFO)
            if (av_frame_side_data_clone(&par->side_data, &par->nb_side_data, sd, 0) < 0)
                fail("out of memory", 0);
    }
    if ((ret = avfilter_graph_create_filter(&g->src, avfilter_get_by_name("abuffer"), "in", args,
                                            NULL, g->graph)) < 0 ||
        (ret = av_buffersrc_parameters_set(g->src, par)) < 0 ||
        (ret = avfilter_graph_create_filter(&format, avfilter_get_by_name("aformat"), "format",
                                            "sample_fmts=flt:sample_rates=16000:channel_layouts=mono",
                                            NULL, g->graph)) < 0 ||
        (ret = avfilter_graph_create_filter(&g->sink, avfilter_get_by_name("abuffersink"), "out",
                                            NULL, NULL, g->graph)) < 0 ||
        (ret = avfilter_link(g->src, 0, format, 0)) < 0 ||
        (ret = avfilter_link(format, 0, g->sink, 0)) < 0 ||
        (ret = avfilter_graph_config(g->graph, NULL)) < 0)
        fail("filter graph", ret);
    av_frame_side_data_free(&par->side_data, &par->nb_side_data);
    av_free(par);
    g->format = f->format;
    g->rate = f->sample_rate;
    if (av_channel_layout_copy(&g->layout, &f->ch_layout) < 0)
        fail("out of memory", 0);
    g->downmix = !!dmx;
    if (dmx)
        memcpy(&g->downmix_info, dmx->data, sizeof(g->downmix_info));
    g->in = 0;
    g->out = out;
}

static void graph_close(Graph *g)
{
    avfilter_graph_free(&g->graph);
    av_channel_layout_uninit(&g->layout);
}

/* What makes the CLI rebuild its filter graph (send_frame() in fftools/ffmpeg_filter.c). */
static int graph_changed(const Graph *g, const AVFrame *f)
{
    const AVFrameSideData *dmx = av_frame_get_side_data(f, AV_FRAME_DATA_DOWNMIX_INFO);

    return g->format != f->format || g->rate != f->sample_rate ||
           av_channel_layout_compare(&g->layout, &f->ch_layout) || g->downmix != !!dmx ||
           (dmx && memcmp(dmx->data, &g->downmix_info, sizeof(g->downmix_info)));
}

static void graph_drain(Graph *g, AVFrame *out)
{
    int ret;

    while ((ret = av_buffersink_get_frame(g->sink, out)) >= 0) {
        put(g->out, (const float *)out->data[0], out->nb_samples);
        g->out += out->nb_samples;
        av_frame_unref(out);
    }
    if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
        fail("filter", ret);
}

/* pts are a running sample count: aresample does not compensate, so they don't touch samples. */
static void graph_push(Graph *g, AVFrame *f, AVFrame *out)
{
    int ret;

    f->pts = g->in;
    f->time_base = (AVRational){1, f->sample_rate};
    g->in += f->nb_samples;
    if ((ret = av_buffersrc_add_frame_flags(g->src, f, AV_BUFFERSRC_FLAG_PUSH)) < 0)
        fail("filter", ret);
    graph_drain(g, out);
}

/* At the end of the stream the resampler's buffered tail is flushed, as by the CLI. */
static void graph_end(Graph *g, AVFrame *out)
{
    int ret;

    if ((ret = av_buffersrc_add_frame_flags(g->src, NULL, 0)) < 0)
        fail("filter", ret);
    graph_drain(g, out);
}

/* --------------------------------------------------------------- full mode -- */

/* Decodes every packet, as the CLI does. */
static void full(const char *path)
{
    AVPacket *pkt = av_packet_alloc();
    AVFrame *f = av_frame_alloc(), *out = av_frame_alloc();
    AVCodecContext *dec;
    Graph g = {0};
    int64_t pos = 0;
    int more, ret;

    if (!pkt || !f || !out)
        fail("out of memory", 0);
    open_input(path);
    dec = decoder_open();
    base = len = 0;
    do {
        more = next_packet(pkt);
        /* Rejected packets and frames are skipped, as by the CLI without -xerror. */
        avcodec_send_packet(dec, more ? pkt : NULL);
        av_packet_unref(pkt);
        while ((ret = avcodec_receive_frame(dec, f)) != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            if (ret < 0)
                continue;
            /* The CLI's rebuild drops what the resampler still holds and continues the output. */
            if (g.graph && graph_changed(&g, f)) {
                pos = g.out;
                graph_close(&g);
            }
            if (!g.graph)
                graph_open(&g, f, pos);
            graph_push(&g, f, out);
        }
    } while (more);
    if (g.graph)
        graph_end(&g, out);
}

/* ----------------------------------------------------------- windowed mode -- */

/*
 * A segment decodes from PREROLL packets before a window's first input sample with a fresh
 * decoder and filter graph. The filter starts at an input sample s0 that is a multiple of the
 * resampling ratio's input side, so the resampler's phase there equals the full decode's, and
 * s0 lies MARGIN output samples before the window. Positions come from pts, which are
 * frame-exact in the containers windowable() admits.
 *
 * Whatever cannot be made exact restarts the file in full mode (longjmp to main): windows
 * already written stay valid, because none of them depends on the packet that triggered it.
 */

static jmp_buf restart;

static int windowable(void)
{
    const char *name = fmt->iformat->name;
    enum AVCodecID id = st->codecpar->codec_id;

    /* Vorbis in Ogg is not here: Ogg misdates some packets, and a page dropped for a bad CRC
     * could not be told apart from that. AAC's noise substitution runs across the whole file;
     * Opus and WMA are not exact from a fresh decoder. */
    return (!strcmp(name, "mp3") &&
            (id == AV_CODEC_ID_MP1 || id == AV_CODEC_ID_MP2 || id == AV_CODEC_ID_MP3)) ||
           (!strcmp(name, "flac") && id == AV_CODEC_ID_FLAC) ||
           ((!strcmp(name, "wav") || !strcmp(name, "aiff")) &&
            id >= AV_CODEC_ID_FIRST_AUDIO && id < AV_CODEC_ID_ADPCM_IMA_QT);
}

typedef struct {
    AVPacket *pkt;
    int64_t idx, at; /* packet number; pts in input samples, origin not yet subtracted */
} Slot;

static Slot ring[RING];
static int rate, ain, aout;    /* input rate; resampling ratio ain:aout in lowest terms */
static int64_t origin = AV_NOPTS_VALUE; /* pts of the full decode's first sample, in samples */
static AVFrame *ref;           /* the first decoded frame's parameters */
static uint32_t mpa;           /* the first MP3 header's version, layer, rate and mono bits */

static struct {
    AVCodecContext *dec;
    Graph g;
    int64_t fed;     /* last packet fed */
    int64_t s0;      /* input sample the filter starts at */
    int zero;        /* started at packet 0: positions are counted, not taken from pts */
    int frames;
    int64_t at0;     /* position of the first frame */
    int64_t decoded; /* samples decoded */
} seg;

static int64_t samples(int64_t pts)
{
    return av_rescale_q(pts, st->time_base, (AVRational){1, rate});
}

/* A packet the decoder would reject drops its samples from the full decode and shifts every
 * later position, but windowed mode never decodes the packets between segments; so they are
 * checked here, and parameter changes, which rebuild the full decode's filter graph, too. */
static void check_packet(const AVPacket *pkt)
{
    const uint8_t *p = pkt->data;
    int n = pkt->size;
    uint32_t h, keep;

    if (pkt->pts == AV_NOPTS_VALUE ||
        av_packet_get_side_data(pkt, AV_PKT_DATA_NEW_EXTRADATA, NULL) ||
        av_packet_get_side_data(pkt, AV_PKT_DATA_PARAM_CHANGE, NULL))
        longjmp(restart, 1);
    switch (st->codecpar->codec_id) {
    case AV_CODEC_ID_FLAC:
        /* The frame CRC-16 covers header and data; flacdec runs it under AV_EF_CRCCHECK. Any
         * corrupt frame restarts, whether the decoder would reject it or not. The flac
         * parser mostly cuts packets differently around a corrupt frame first, which leaves
         * one without pts. */
        if (av_crc(av_crc_get_table(AV_CRC_16_ANSI), 0, p, n))
            longjmp(restart, 1);
        break;
    case AV_CODEC_ID_MP1:
    case AV_CODEC_ID_MP2:
    case AV_CODEC_ID_MP3:
        /* The header checks decode_frame() in libavcodec/mpegaudiodec_template.c rejects a
         * packet on: no sync, reserved version or layer, free format, bitrate 15, sample
         * rate 3. It skips leading zeros first. A frame whose side info it rejects
         * (big_values > 288, block_type 0) is not caught. */
        while (n && !*p)
            p++, n--;
        if (n < 4)
            longjmp(restart, 1);
        h = AV_RB32(p);
        if ((h & 0xffe00000) != 0xffe00000 || (h >> 19 & 3) == 1 || !(h >> 17 & 3) ||
            !(h >> 12 & 15) || (h >> 12 & 15) == 15 || (h >> 10 & 3) == 3)
            longjmp(restart, 1);
        keep = (h & 0x001e0c00) | ((h >> 6 & 3) == 3);
        if (!mpa)
            mpa = keep;
        else if (keep != mpa)
            longjmp(restart, 1);
        break;
    default:
        break;
    }
}

/* Drops a frame's first off samples. */
static void trim(AVFrame *f, int off)
{
    AVFrame *t = av_frame_alloc();

    if (!t || av_frame_copy_props(t, f) < 0 || av_channel_layout_copy(&t->ch_layout, &f->ch_layout) < 0)
        fail("out of memory", 0);
    t->format = f->format;
    t->sample_rate = f->sample_rate;
    t->nb_samples = f->nb_samples - off;
    if (av_frame_get_buffer(t, 0) < 0)
        fail("out of memory", 0);
    av_samples_copy(t->extended_data, f->extended_data, 0, off, t->nb_samples,
                    f->ch_layout.nb_channels, f->format);
    av_frame_unref(f);
    av_frame_move_ref(f, t);
    av_frame_free(&t);
}

static void handle(AVFrame *f, AVFrame *out)
{
    int64_t pos, n = f->nb_samples;

    if (f->format != ref->format || f->sample_rate != ref->sample_rate ||
        av_channel_layout_compare(&f->ch_layout, &ref->ch_layout) ||
        av_frame_get_side_data(f, AV_FRAME_DATA_DOWNMIX_INFO))
        longjmp(restart, 1);
    if (seg.zero)
        pos = seg.decoded;
    else {
        if (seg.frames < CHECKED) {
            if (f->pts == AV_NOPTS_VALUE)
                longjmp(restart, 1);
            if (!seg.frames)
                seg.at0 = samples(f->pts) - origin;
            else if (samples(f->pts) - origin - seg.at0 != seg.decoded)
                longjmp(restart, 1);
            /* The checked frames must end before the filter's first sample. */
            if (seg.at0 + seg.decoded + n > seg.s0)
                longjmp(restart, 1);
        }
        pos = seg.at0 + seg.decoded;
    }
    seg.frames++;
    seg.decoded += n;
    if (pos + n <= seg.s0)
        return;
    if (pos < seg.s0)
        trim(f, seg.s0 - pos);
    if (!seg.g.graph)
        graph_open(&seg.g, f, seg.s0 / ain * aout);
    graph_push(&seg.g, f, out);
}

/* A fresh decoder may fail where the full decode does not: decode everything instead. */
static void feed(const AVPacket *pkt, AVFrame *f, AVFrame *out)
{
    int ret;

    if (avcodec_send_packet(seg.dec, pkt) < 0 && pkt)
        longjmp(restart, 1);
    while ((ret = avcodec_receive_frame(seg.dec, f)) != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
        if (ret < 0)
            longjmp(restart, 1);
        handle(f, out);
    }
}

static void seg_close(void)
{
    avcodec_free_context(&seg.dec);
    if (seg.g.graph)
        graph_close(&seg.g);
}

/* The input sample the next window's segment starts the filter at. */
static int64_t target(void)
{
    int64_t m0 = window() - MARGIN;

    return (m0 <= 0 ? 0 : m0 / aout) * ain;
}

/* Feeds the current segment, or starts a new one, as far as the next window needs packets up
 * to i. Until a packet past the window's first input sample has been read, it is not known
 * where that segment starts, so packets wait in the ring. */
static void advance(int64_t i, AVFrame *f, AVFrame *out)
{
    while (!seg.dec || seg.fed < i) {
        int64_t s0 = target(), j, start;

        for (j = FFMAX(i - RING + 1, 0); j <= i && ring[j % RING].at - origin <= s0; j++)
            ;
        if (j > i) {
            /* Not continued past a gap longer than the preroll: the next segment starts later. */
            if (seg.dec && i - seg.fed > PREROLL + 1)
                seg_close();
            return;
        }
        start = j - 1 - PREROLL;
        if (!seg.dec || start > seg.fed + 1) {
            if (seg.dec)
                seg_close();
            start = FFMAX(start, 0);
            seg.dec = decoder_open();
            seg.fed = start - 1;
            seg.s0 = s0;
            seg.zero = !start;
            seg.frames = 0;
            seg.decoded = 0;
        }
        while (seg.fed < i) {
            Slot *s = &ring[(seg.fed + 1) % RING];

            if (s->idx != seg.fed + 1)
                longjmp(restart, 1);
            feed(s->pkt, f, out);
            seg.fed++;
        }
    }
}

static void windowed(void)
{
    AVPacket *pkt = av_packet_alloc();
    AVFrame *f = av_frame_alloc(), *out = av_frame_alloc();
    AVCodecContext *probe;
    int64_t i, g;
    int ret;

    if (!pkt || !f || !out || !(ref = av_frame_alloc()))
        fail("out of memory", 0);
    if ((rate = st->codecpar->sample_rate) <= 0)
        longjmp(restart, 1);
    g = av_gcd(rate, OUT_RATE);
    ain = rate / g;
    aout = OUT_RATE / g;
    for (i = 0; i < RING; i++) {
        if (!(ring[i].pkt = av_packet_alloc()))
            fail("out of memory", 0);
        ring[i].idx = -1;
    }
    probe = decoder_open();
    for (i = 0; next_packet(pkt); i++) {
        Slot *s = &ring[i % RING];

        check_packet(pkt);
        av_packet_unref(s->pkt);
        av_packet_move_ref(s->pkt, pkt);
        s->idx = i;
        s->at = samples(s->pkt->pts);
        if (origin == AV_NOPTS_VALUE) {
            /* The full decode's first sample: the first frame decoded from packet 0. */
            if (avcodec_send_packet(probe, s->pkt) < 0)
                longjmp(restart, 1);
            if ((ret = avcodec_receive_frame(probe, f)) >= 0) {
                if (f->pts == AV_NOPTS_VALUE || f->sample_rate != rate)
                    longjmp(restart, 1);
                origin = samples(f->pts);
                av_frame_move_ref(ref, f);
                avcodec_free_context(&probe);
            } else if (ret != AVERROR(EAGAIN))
                longjmp(restart, 1);
            else if (i >= PREROLL) /* packet 0 must stay in the ring */
                longjmp(restart, 1);
            else
                continue;
        }
        advance(i, f, out);
    }
    if (seg.dec && seg.fed == i - 1) {
        /* The segment reached the end: flush decoder and filter, as the CLI does. */
        feed(NULL, f, out);
        if (seg.g.graph)
            graph_end(&seg.g, out);
    } else if (i && origin != AV_NOPTS_VALUE) {
        /* The next window's segment never started: it begins after the last packet's pts.
         * Only a last packet longer than a window could still fill it. */
        Slot *s = &ring[(i - 1) % RING];

        if (s->at - origin + samples(s->pkt->duration) > target() + (int64_t)span * ain / aout)
            longjmp(restart, 1);
    }
}

int main(int argc, char **argv)
{
    av_log_set_level(AV_LOG_QUIET);
    if (argc == 3 && !strcmp(argv[1], "hash"))
        return hash(argv[2]);
    if (argc != 6 || strcmp(argv[1], "windows"))
        fail("usage: sonic-decode hash FILE | sonic-decode windows FILE FIRST STEP SPAN", 0);
    first = strtoll(argv[3], NULL, 10);
    step = strtoll(argv[4], NULL, 10);
    span = atoi(argv[5]);
    if (step <= 0 || span <= 0 || first < -span)
        fail("bad window layout", 0);
    if (!(zeros = calloc(span, sizeof(*zeros))))
        fail("out of memory", 0);
    setvbuf(stdout, NULL, _IOFBF, 1 << 16);
    open_input(argv[2]);
    if (!windowable())
        full(argv[2]);
    else if (setjmp(restart))
        full(argv[2]);
    else
        windowed();
    return fflush(stdout) || ferror(stdout);
}

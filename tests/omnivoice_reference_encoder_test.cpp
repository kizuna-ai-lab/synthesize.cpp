// The HuBERT semantic branch and the codec's own SemanticEncoder conv stack,
// compared at a synthetic miniature scale against the REAL transformers
// classes (HubertModel and, from modeling_higgs_audio_v2_tokenizer.py, the
// real SemanticEncoder / HiggsAudioV2TokenizerSemanticEncoderBlock /
// HiggsAudioV2TokenizerResidualUnit) run by
// scripts/dump_reference_omnivoice_semantic.py.
//
// Three things are pinned here, each isolating a stage the others could hide
// a bug in:
//
// `kExpectedMean` -- the mean over all `layer_count + 1` hidden states,
// BEFORE the [::2] downsample. This is architecturally what the oracle's own
// `ref/semantic_mean.f32` probe captures (see
// scripts/dump_reference_omnivoice_pytorch.py's SemanticMeanProbe), so a
// real-scale replay mismatch can be triangulated against this miniature
// check before it is ever attributed to a placement or numerics difference
// specific to real scale.
//
// `kExpectedDownsampled` -- the same mean after taking every other position,
// which isolates the stride-2 slice from everything feeding it and
// everything that reads it.
//
// `kExpectedFinal` -- the SemanticEncoder's own output, which is the only
// thing here that exercises the codec's res-unit/ELU/biased-exit-conv stack
// at all.
//
// The positional convolution's weight-norm parametrization is folded on the
// Python side (torch._weight_norm, the same operator
// scripts/convert-omnivoice.py's fold_weight_norm calls on the real
// checkpoint) before the dump script ever prints a fixture array, so
// `kPosConvWeight` below is already the plain kernel the catalog resolves --
// this test never reconstructs the parametrized pair. The LCG stream still
// draws the magnitude and direction tensors that fold INTO that kernel
// (see fill_hubert), because skipping those draws would desynchronize every
// later value from the Python script's stream at the exact same point.
//
// Weights are not carried in either file. Both sides draw them from the same
// 64-bit LCG in the same order, so only the outputs are pinned.

#include "arch/omnivoice/catalog.h"
#include "arch/omnivoice/reference-encoder.h"
#include "arch/omnivoice/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "test-assert.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace {

// The miniature HuBERT scale (scripts/dump_reference_omnivoice_semantic.py's
// own constants -- CONV_DIM/CONV_KERNEL/CONV_STRIDE/HIDDEN/HEADS/INTERMEDIATE/
// LAYERS/POS_KERNEL/POS_GROUPS/PCM_SAMPLES): three feature-extractor layers
// (layer 0 keeps the real checkpoint's GroupNorm structure, every layer keeps
// its stride-2 pattern), two attention layers, an even positional-convolution
// kernel (exercises HubertSamePadLayer's trim, like the real checkpoint's
// even 128).
constexpr uint32_t kConvDim[3]    = { 4, 4, 4 };
constexpr uint32_t kConvKernel[3] = { 3, 3, 2 };
constexpr uint32_t kConvStride[3] = { 2, 2, 2 };
constexpr uint32_t kHidden        = 8;
constexpr uint32_t kHeads         = 2;
constexpr uint32_t kIntermediate  = 12;
constexpr uint32_t kLayers        = 2;
constexpr uint32_t kPosKernel     = 4;
constexpr uint32_t kPosGroups     = 2;
constexpr uint32_t kPosInPerGroup = kHidden / kPosGroups;
constexpr float    kLayerNormEps  = 1e-5f;
constexpr uint32_t kPcmSamples    = 8;

// codec.encoder_semantic, already minimal at real scale: two stride-1 blocks
// of two dilation-1 residual units, unchanged from the real checkpoint.
constexpr size_t kSemBlocks = 2;
constexpr size_t kSemUnits  = 2;

// Measured worst case on CPU: 5.96e-7 (mean 4.77e-7, downsampled 4.17e-7,
// final 5.96e-7) -- observed x5, rather than a loose round number, so a
// regression that is still an order of magnitude away from FP32 noise still
// trips this. No accelerator is exercised in this environment to measure
// against; kAcceleratorTolerance keeps omnivoice_codec_test.cpp's own
// TF32-aware bound rather than inventing an unmeasured one.
constexpr float kCpuTolerance         = 3e-6f;
constexpr float kAcceleratorTolerance = 5e-3f;

constexpr float kWeightScale = 0.3f, kWeightOffset = 0.0f;
constexpr float kBiasScale = 0.1f, kBiasOffset = 0.0f;
constexpr float kGainScale = 0.25f, kGainOffset = 1.0f;

// values from scripts/dump_reference_omnivoice_semantic.py, pasted verbatim,
// at transformers 5.14.1, torch 2.13.0+cu130. That run's own shape report:
// mean (40, 8), downsampled (20, 8), final (8, 20).

constexpr float kPcm16k[] = {
    -0.0177596807f, -0.112141669f, -0.231427431f, 0.391185403f,
    -0.389479876f,  -0.451198459f, 0.127516568f,  -0.195634663f,
};

// Already the folded plain kernel (torch._weight_norm(direction, magnitude,
// dim=2)), ne = [kernel, in_per_group, hidden] once reversed into ggml order.
constexpr float kPosConvWeight[] = {
    0.0584841669f,  0.299363673f,    0.157314792f,   -0.160484701f,   0.0147981895f,   0.00284637185f,  0.225702316f,
    -0.159273043f,  0.2138412f,      -0.256452888f,  0.000768830185f, 0.336535007f,    -0.00128162163f, 0.153507084f,
    -0.0573827401f, 0.160809502f,    0.412859797f,   -0.191518158f,   -0.283147871f,   -0.355509996f,   0.299947917f,
    -0.158041224f,  -0.135582626f,   -0.106090091f,  -0.279238969f,   -0.255169332f,   0.2269907f,      -0.337664932f,
    0.205488771f,   -0.0222886316f,  0.179511487f,   0.176645845f,    0.112983108f,    0.194881648f,    -0.130341932f,
    0.111450762f,   0.0202239621f,   0.0687684119f,  -0.322663903f,   0.213174358f,    -0.272539914f,   -0.303913116f,
    0.186163485f,   -0.286494642f,   -0.278473347f,  -0.0326853991f,  -0.215024561f,   0.0863702819f,   0.0279517584f,
    -0.130604029f,  0.0671657547f,   -0.341941148f,  -0.34358269f,    0.24517867f,     0.241395041f,    0.0349982344f,
    0.0188644789f,  0.0341163576f,   -0.0269170552f, -0.245549873f,   0.208333746f,    0.200791806f,    0.300110996f,
    -0.0643830523f, 0.153092459f,    0.0734393075f,  -0.245568469f,   0.0130685195f,   0.345866382f,    0.297735333f,
    0.0714560375f,  -0.356446803f,   -0.071058549f,  0.321668178f,    -0.299334556f,   -0.0321462341f,  -0.125173271f,
    -0.193012938f,  0.174929738f,    0.367230475f,   0.0568589307f,   0.331797063f,    -0.244279668f,   -0.045492813f,
    -0.232520685f,  -0.205421582f,   -0.223097488f,  0.198836267f,    -0.0345569141f,  0.164506227f,    0.20398818f,
    -0.047046002f,  -0.0066504986f,  0.0671340004f,  0.0371848308f,   -0.00911394227f, 0.00623701466f,  -0.316002697f,
    0.0229122173f,  -0.259707808f,   0.0699765235f,  0.0138080632f,   -0.324318409f,   -0.0766830295f,  -0.133396491f,
    0.0800302997f,  0.278182179f,    0.105676867f,   -0.423190773f,   0.26402688f,     0.13498041f,     -0.144600973f,
    0.0488108322f,  -0.166696474f,   -0.190160632f,  0.318339944f,    0.162751645f,    0.153330222f,    -0.30198127f,
    -0.119862333f,  -0.00211184635f, -0.160691574f,  0.0358795822f,   -0.193185791f,   0.385996073f,    -0.0549174249f,
    -0.128015161f,  0.355437994f,
};

constexpr float kExpectedMean[] = {
    -1.45657551f, 0.469723135f, -1.37439668f,  -0.540444553f, -0.0927879512f, 0.953898609f, 0.617733836f, 1.20959079f,
    -1.52127492f, 0.916015625f, -1.27620649f,  -0.666300058f, -0.0700652003f, 0.885292768f, 0.519263566f, 1.00958109f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.67447519f, 0.803821385f, -1.05880606f,  -0.642365277f, -0.0365027674f, 0.834845483f, 0.35063687f,  1.2612282f,
    -1.64295447f, 0.766293943f, -0.802588284f, -1.04138219f,  0.518274486f,   1.01223898f,  0.12259493f,  0.905781269f,
    -1.46529114f, 0.425762564f, -1.26092625f,  -0.585840821f, -0.183262125f,  0.662339985f, 0.730743706f, 1.53766668f,
    -1.7814306f,  0.597802818f, -0.920883596f, -0.821477234f, 0.220897004f,   0.807390749f, 0.629753411f, 1.10102272f,
    -1.6651901f,  0.86470598f,  -0.99553889f,  -0.655031562f, 0.0122626526f,  0.739233077f, 0.17234087f,  1.41037476f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f,  -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.45821464f, 0.149270341f, -1.07043588f,  -0.969615042f, 0.254546285f,   0.828025997f, 0.713642538f, 1.40321481f,
};

constexpr float kExpectedDownsampled[] = {
    -1.45657551f, 0.469723135f, -1.37439668f, -0.540444553f, -0.0927879512f, 0.953898609f, 0.617733836f, 1.20959079f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.67447519f, 0.803821385f, -1.05880606f, -0.642365277f, -0.0365027674f, 0.834845483f, 0.35063687f,  1.2612282f,
    -1.46529114f, 0.425762564f, -1.26092625f, -0.585840821f, -0.183262125f,  0.662339985f, 0.730743706f, 1.53766668f,
    -1.6651901f,  0.86470598f,  -0.99553889f, -0.655031562f, 0.0122626526f,  0.739233077f, 0.17234087f,  1.41037476f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
    -1.68132246f, 0.747687876f, -1.03663564f, -0.721013069f, 0.0640524104f,  0.813710272f, 0.390472889f, 1.27071202f,
};

// Already [T'', HIDDEN]: dumped as final.T (the codec script's own "C++
// read-back order" convention), since SemanticEncoder's raw output is
// channel-first ([HIDDEN, T'']), the opposite of this file's [HIDDEN, T'']
// ggml convention's natural flatten order.
constexpr float kExpectedFinal[] = {
    -0.0514649078f, -0.35378933f,   0.27806282f,    -0.395775467f,   0.323000789f,   -0.470552683f,   -0.00568839908f,
    -0.268567622f,  -0.769493282f,  -0.653374434f,  0.748015285f,    0.225219637f,   0.311307192f,    -0.625151098f,
    -0.373842418f,  -0.880164444f,  -0.33914575f,   -0.827468574f,   0.456812918f,   -0.111600004f,   0.261237293f,
    -0.779423118f,  -1.14285648f,   -0.872093141f,  -0.561060429f,   -0.737589836f,  0.49846828f,     -0.071296297f,
    0.331020832f,   -0.868046343f,  -1.26456976f,   -0.946988881f,   -0.572015166f,  -0.698117971f,   0.555281103f,
    -0.0431473404f, 0.413113892f,   -0.848273695f,  -1.25606596f,    -0.902842402f,  -0.532467782f,   -0.693775058f,
    0.54787147f,    -0.0443333611f, 0.412134588f,   -0.83861357f,    -1.25851786f,   -0.909776747f,   -0.534216344f,
    -0.698320746f,  0.551694572f,   -0.0400855243f, 0.415739894f,    -0.832934201f,  -1.27943122f,    -0.903366387f,
    -0.539786875f,  -0.690926969f,  0.561797798f,   -0.0758401006f,  0.392386317f,   -0.816288769f,   -1.26363242f,
    -0.881559789f,  -0.509372115f,  -0.677621841f,  0.598826647f,    -0.0148144159f, 0.422729373f,    -0.79873991f,
    -1.24293792f,   -0.926804483f,  -0.511504412f,  -0.641719759f,   0.460684955f,   -0.00684249308f, 0.493896961f,
    -0.870271981f,  -1.19828427f,   -0.859243035f,  -0.513406038f,   -0.517274201f,  0.540986001f,    0.0337548181f,
    0.53921771f,    -0.940956235f,  -1.24611294f,   -0.958407283f,   -0.462186933f,  -0.684826434f,   0.570318758f,
    -0.0902462825f, 0.244016185f,   -0.856980562f,  -1.12697232f,    -0.939545929f,  -0.555941224f,   -0.699178696f,
    0.514677346f,   -0.104271591f,  0.209485278f,   -0.821807861f,   -1.30400932f,   -0.937081993f,   -0.534046292f,
    -0.810613751f,  0.547222257f,   -0.135120049f,  0.384858012f,    -0.875312328f,  -1.30987632f,    -0.840101182f,
    -0.544041812f,  -0.696784973f,  0.595585108f,   -0.00546490587f, 0.439685583f,   -0.835013866f,   -1.23453867f,
    -0.907592773f,  -0.527991056f,  -0.692729831f,  0.557712495f,    -0.0353499614f, 0.411869764f,    -0.837157011f,
    -1.2462939f,    -0.92411685f,   -0.55233252f,   -0.669046819f,   0.536343753f,   -0.00269116834f, 0.458704531f,
    -0.895681083f,  -1.22365975f,   -0.905802429f,  -0.607514083f,   -0.620342433f,  0.554266572f,    -0.0559307151f,
    0.406790018f,   -0.923120618f,  -1.23433495f,   -0.873809159f,   -0.591194332f,  -0.710045218f,   0.784410775f,
    0.0618086159f,  0.40624845f,    -0.941136956f,  -1.11066306f,    -0.72959739f,   -0.163449511f,   -0.266589433f,
    0.655845165f,   0.371414155f,   0.271875501f,   -0.304496437f,   -1.58000088f,   -0.651540041f,
};

// The twin of LcgStream in the reference script. Every value is a 24-bit
// numerator over 2^23, so both sides start from bit-identical inputs.
class LcgStream {
  public:
    explicit LcgStream(uint64_t seed) : state_(seed) {}

    float next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return float(state_ >> 40) / 8388608.0f - 1.0f;
    }

    std::vector<float> fill(size_t count, float scale, float offset) {
        std::vector<float> values(count);
        for (float & value : values) {
            value = next() * scale + offset;
        }
        return values;
    }

  private:
    uint64_t state_;
};

constexpr uint64_t kSeed = 20260801u;

struct ContextDeleter {
    void operator()(ggml_context * context) const {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

using Context = std::unique_ptr<ggml_context, ContextDeleter>;

// Generous: check_rejections alone builds the full graph successfully once
// and then partially (HuBERT branch complete, encoder_semantic refused) a
// second time, sharing one context across every sub-case the way
// omnivoice_codec_test.cpp's check_rejections does for its own valid builds.
constexpr size_t kNodeBudget = 4096;

Context make_context(size_t bytes) {
    ggml_init_params parameters{};
    parameters.mem_size = bytes;
    parameters.no_alloc = true;
    return Context(ggml_init(parameters));
}

Context make_graph_context() {
    return make_context(ggml_tensor_overhead() * (kNodeBudget + 64) + ggml_graph_overhead_custom(kNodeBudget, false));
}

synth::omnivoice::HParams make_hparams() {
    synth::omnivoice::HParams hparams;
    hparams.semantic.hidden_size          = kHidden;
    hparams.semantic.layer_count          = kLayers;
    hparams.semantic.attention_head_count = kHeads;
    hparams.semantic.intermediate_size    = kIntermediate;
    hparams.semantic.layer_norm_eps       = kLayerNormEps;
    hparams.semantic.conv_dim.assign(std::begin(kConvDim), std::end(kConvDim));
    hparams.semantic.conv_kernel.assign(std::begin(kConvKernel), std::end(kConvKernel));
    hparams.semantic.conv_stride.assign(std::begin(kConvStride), std::end(kConvStride));
    return hparams;
}

struct Fixture {
    ggml_backend_t                 backend = nullptr;
    Context                        persistent;
    ggml_backend_buffer_t          buffer = nullptr;
    synth::omnivoice::ModelWeights weights;
    ggml_tensor *                  pcm = nullptr;

    Fixture()                            = default;
    Fixture(const Fixture &)             = delete;
    Fixture & operator=(const Fixture &) = delete;

    ~Fixture() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

bool build_fixture(ggml_backend_dev_t device, Fixture & fixture) {
    fixture.backend = ggml_backend_dev_init(device, nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }
    fixture.persistent  = make_context(ggml_tensor_overhead() * 256);
    ggml_context * pctx = fixture.persistent.get();
    if (pctx == nullptr) {
        return false;
    }

    // `ordered` mirrors the reference script's per-tensor draw sequence.
    // Every real tensor carries `skip_count == 0` and is drawn+set at its own
    // element count; a `nullptr` entry with `skip_count > 0` is a draw the
    // Python side made but this side does not need the VALUE of (the
    // positional convolution's weight-norm pair, folded away below) -- it
    // still has to happen, at the same count, to keep the two streams
    // synchronized from that point on.
    std::vector<ggml_tensor *> ordered;
    std::vector<float>         scales;
    std::vector<float>         offsets;
    std::vector<size_t>        skip_counts;
    auto                       add = [&](ggml_tensor * tensor, float scale, float offset) {
        ordered.push_back(tensor);
        scales.push_back(scale);
        offsets.push_back(offset);
        skip_counts.push_back(0);
        return tensor;
    };
    auto weight = [&](ggml_tensor * tensor) {
        return add(tensor, kWeightScale, kWeightOffset);
    };
    auto bias = [&](ggml_tensor * tensor) {
        return add(tensor, kBiasScale, kBiasOffset);
    };
    auto gain = [&](ggml_tensor * tensor) {
        return add(tensor, kGainScale, kGainOffset);
    };
    auto skip = [&](size_t count) {
        ordered.push_back(nullptr);
        scales.push_back(0.0f);
        offsets.push_back(0.0f);
        skip_counts.push_back(count);
    };

    auto bare_conv = [&](synth::omnivoice::Conv1dWeights & target, int64_t kernel, int64_t in, int64_t out) {
        target.weight = weight(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kernel, in, out));
    };
    auto biased_conv = [&](synth::omnivoice::Conv1dWeights & target, int64_t kernel, int64_t in, int64_t out) {
        target.weight = weight(ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kernel, in, out));
        target.bias   = bias(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, out));
    };
    auto linear = [&](synth::omnivoice::LinearWeights & target, int64_t in, int64_t out) {
        target.weight = weight(ggml_new_tensor_2d(pctx, GGML_TYPE_F32, in, out));
        target.bias   = bias(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, out));
    };
    auto norm = [&](synth::omnivoice::LayerNormWeights & target, int64_t channels) {
        target.weight = gain(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, channels));
        target.bias   = bias(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, channels));
    };

    // The fill order is the reference script's fill_hubert/fill_semantic_encoder,
    // module by module in forward order; reordering any line silently changes
    // every weight after it.
    synth::omnivoice::SemanticModelWeights & hubert = fixture.weights.semantic_model;
    hubert.feat_conv.resize(std::size(kConvDim));
    for (size_t index = 0; index < hubert.feat_conv.size(); ++index) {
        const int64_t in = index == 0 ? 1 : int64_t(kConvDim[index - 1]);
        bare_conv(hubert.feat_conv[index], kConvKernel[index], in, kConvDim[index]);
    }
    norm(hubert.feat_conv_norm, kConvDim[0]);
    norm(hubert.feature_projection_norm, kConvDim[std::size(kConvDim) - 1]);
    linear(hubert.feature_projection, kConvDim[std::size(kConvDim) - 1], kHidden);

    // The positional convolution's weight-norm pair (magnitude, then
    // direction) is drawn here to keep the LCG stream synchronized with the
    // reference script's fill_hubert, but neither value is used: the C++
    // graph reads the already-folded plain kernel below (kPosConvWeight),
    // exactly as the catalog resolves it from a converted package.
    skip(kPosKernel);                                     // magnitude, original0: [1, 1, kPosKernel]
    skip(size_t(kHidden) * kPosInPerGroup * kPosKernel);  // direction, original1: [kHidden, kPosInPerGroup, kPosKernel]
    hubert.pos_conv.weight = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, kPosKernel, kPosInPerGroup, kHidden);
    hubert.pos_conv.bias   = bias(ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kHidden));

    hubert.layers.resize(kLayers);
    for (synth::omnivoice::SemanticLayerWeights & layer : hubert.layers) {
        linear(layer.q_proj, kHidden, kHidden);
        linear(layer.k_proj, kHidden, kHidden);
        linear(layer.v_proj, kHidden, kHidden);
        linear(layer.out_proj, kHidden, kHidden);
        norm(layer.layer_norm, kHidden);
        linear(layer.inter_dense, kHidden, kIntermediate);
        linear(layer.output_dense, kIntermediate, kHidden);
        norm(layer.final_layer_norm, kHidden);
    }
    norm(hubert.encoder_norm, kHidden);

    synth::omnivoice::SemanticEncoderWeights & sem_enc = fixture.weights.encoder_semantic;
    bare_conv(sem_enc.conv, 3, kHidden, kHidden);
    sem_enc.blocks.resize(kSemBlocks);
    for (synth::omnivoice::SemanticEncoderBlock & block : sem_enc.blocks) {
        block.res_units.resize(kSemUnits);
        for (synth::omnivoice::SemanticEncoderResUnit & unit : block.res_units) {
            bare_conv(unit.conv1, 3, kHidden, kHidden);
            bare_conv(unit.conv2, 1, kHidden, kHidden);
        }
        biased_conv(block.conv, 3, kHidden, kHidden);
    }

    fixture.pcm = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, kPcmSamples);

    fixture.buffer = ggml_backend_alloc_ctx_tensors(pctx, fixture.backend);
    if (fixture.buffer == nullptr) {
        return false;
    }

    LcgStream stream(kSeed);
    for (size_t index = 0; index < ordered.size(); ++index) {
        if (ordered[index] == nullptr) {
            stream.fill(skip_counts[index], scales[index], offsets[index]);  // draw and discard
            continue;
        }
        const std::vector<float> values =
            stream.fill(size_t(ggml_nelements(ordered[index])), scales[index], offsets[index]);
        ggml_backend_tensor_set(ordered[index], values.data(), 0, ggml_nbytes(ordered[index]));
    }
    // The folded kernel itself, never drawn from the stream.
    ggml_backend_tensor_set(hubert.pos_conv.weight, kPosConvWeight, 0, ggml_nbytes(hubert.pos_conv.weight));

    // The reference script draws PCM_SAMPLES more values here
    // (scripts/dump_reference_omnivoice_semantic.py's `pcm = stream.fill(...)`
    // call), but nothing on either side reads the stream again afterward, so
    // this side skips straight to setting the dumped result (kPcm16k is that
    // same draw, printed verbatim) rather than repeating a call whose output
    // would only be thrown away.
    ggml_backend_tensor_set(fixture.pcm, kPcm16k, 0, ggml_nbytes(fixture.pcm));
    return true;
}

bool compute(Fixture &                          fixture,
             ggml_cgraph *                      graph,
             const std::vector<ggml_tensor *> & outputs,
             std::vector<std::vector<float>> &  values) {
    for (ggml_tensor * output : outputs) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(fixture.backend));
    bool           ok        = allocator != nullptr && ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ok = ggml_backend_graph_compute(fixture.backend, graph) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        values.clear();
        for (ggml_tensor * output : outputs) {
            std::vector<float> one(size_t(ggml_nelements(output)));
            ggml_backend_tensor_get(output, one.data(), 0, ggml_nbytes(output));
            values.push_back(std::move(one));
        }
    }
    if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
    }
    return ok;
}

float deviation(const std::vector<float> & got, const float * expected, size_t count) {
    if (got.size() != count) {
        return std::numeric_limits<float>::infinity();
    }
    float worst = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        worst = std::fmax(worst, std::fabs(got[index] - expected[index]));
    }
    return worst;
}

// The whole branch, with the mean and downsampled taps probed alongside the
// final SemanticEncoder output so a failure names a stage rather than "the
// output is wrong".
bool run_case(ggml_backend_dev_t device, float & max_diff) {
    Fixture fixture;
    if (!build_fixture(device, fixture)) {
        return false;
    }

    Context       graph_ctx = make_graph_context();
    ggml_cgraph * graph     = ggml_new_graph_custom(graph_ctx.get(), kNodeBudget, false);

    ggml_tensor * mean        = nullptr;
    ggml_tensor * downsampled = nullptr;
    ggml_tensor * final_out   = synth::omnivoice::build_semantic_branch(graph_ctx.get(), fixture.pcm, fixture.weights,
                                                                        make_hparams(), nullptr, &mean, &downsampled);
    if (final_out == nullptr || mean == nullptr || downsampled == nullptr) {
        std::printf("  build_semantic_branch refused to build\n");
        return false;
    }
    if (mean->ne[0] != int64_t(kHidden) || downsampled->ne[0] != int64_t(kHidden) ||
        final_out->ne[0] != int64_t(kHidden)) {
        std::printf("  unexpected hidden width: mean %lld, downsampled %lld, final %lld\n", (long long) mean->ne[0],
                    (long long) downsampled->ne[0], (long long) final_out->ne[0]);
        return false;
    }

    std::vector<std::vector<float>> values;
    if (!compute(fixture, graph, { mean, downsampled, final_out }, values)) {
        return false;
    }
    const float d_mean        = deviation(values[0], kExpectedMean, std::size(kExpectedMean));
    const float d_downsampled = deviation(values[1], kExpectedDownsampled, std::size(kExpectedDownsampled));
    const float d_final       = deviation(values[2], kExpectedFinal, std::size(kExpectedFinal));
    std::printf("  mean %.3g  downsampled %.3g  final %.3g\n", double(d_mean), double(d_downsampled), double(d_final));
    max_diff = std::fmax(std::fmax(d_mean, d_downsampled), d_final);
    return true;
}

// A shape the builder cannot serve is a wiring defect, so it returns nullptr
// rather than aborting inside ggml on an assertion the caller cannot catch.
int check_rejections() {
    Context        context = make_graph_context();
    ggml_context * ctx     = context.get();
    SYNTH_TEST_CHECK(ctx != nullptr);

    synth::omnivoice::ModelWeights weights;
    synth::omnivoice::HParams      hparams = make_hparams();

    auto bare_conv = [&](int64_t kernel, int64_t in, int64_t out) {
        synth::omnivoice::Conv1dWeights target;
        target.weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel, in, out);
        return target;
    };
    auto linear = [&](int64_t in, int64_t out) {
        synth::omnivoice::LinearWeights target;
        target.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, out);
        target.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
        return target;
    };
    auto norm = [&](int64_t channels) {
        synth::omnivoice::LayerNormWeights target;
        target.weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        target.bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
        return target;
    };

    synth::omnivoice::SemanticModelWeights & hubert = weights.semantic_model;
    for (size_t index = 0; index < std::size(kConvDim); ++index) {
        const int64_t in = index == 0 ? 1 : int64_t(kConvDim[index - 1]);
        hubert.feat_conv.push_back(bare_conv(kConvKernel[index], in, kConvDim[index]));
    }
    hubert.feat_conv_norm          = norm(kConvDim[0]);
    hubert.feature_projection_norm = norm(kConvDim[std::size(kConvDim) - 1]);
    hubert.feature_projection      = linear(kConvDim[std::size(kConvDim) - 1], kHidden);
    hubert.pos_conv.weight         = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kPosKernel, kPosInPerGroup, kHidden);
    hubert.pos_conv.bias           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    hubert.layers.resize(kLayers);
    for (synth::omnivoice::SemanticLayerWeights & layer : hubert.layers) {
        layer.q_proj           = linear(kHidden, kHidden);
        layer.k_proj           = linear(kHidden, kHidden);
        layer.v_proj           = linear(kHidden, kHidden);
        layer.out_proj         = linear(kHidden, kHidden);
        layer.layer_norm       = norm(kHidden);
        layer.inter_dense      = linear(kHidden, kIntermediate);
        layer.output_dense     = linear(kIntermediate, kHidden);
        layer.final_layer_norm = norm(kHidden);
    }
    hubert.encoder_norm = norm(kHidden);

    synth::omnivoice::SemanticEncoderWeights & sem_enc = weights.encoder_semantic;
    sem_enc.conv                                       = bare_conv(3, kHidden, kHidden);
    sem_enc.blocks.resize(kSemBlocks);
    for (synth::omnivoice::SemanticEncoderBlock & block : sem_enc.blocks) {
        block.res_units.resize(kSemUnits);
        for (synth::omnivoice::SemanticEncoderResUnit & unit : block.res_units) {
            unit.conv1 = bare_conv(3, kHidden, kHidden);
            unit.conv2 = bare_conv(1, kHidden, kHidden);
        }
        block.conv      = bare_conv(3, kHidden, kHidden);
        block.conv.bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden);
    }

    ggml_tensor * pcm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kPcmSamples);
    SYNTH_TEST_CHECK(synth::omnivoice::build_semantic_branch(ctx, pcm, weights, hparams) != nullptr);

    // A 2-D "pcm" is not the 1-D waveform this builder documents; nothing
    // downstream would notice a stray extra axis until ggml aborted on it.
    ggml_tensor * pcm_2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kPcmSamples, 2);
    SYNTH_TEST_CHECK(synth::omnivoice::build_semantic_branch(ctx, pcm_2d, weights, hparams) == nullptr);

    // A head count that does not divide the hidden width evenly has no
    // well-defined head_dim.
    synth::omnivoice::HParams bad_heads     = hparams;
    bad_heads.semantic.attention_head_count = 3;
    SYNTH_TEST_CHECK(synth::omnivoice::build_semantic_branch(ctx, pcm, weights, bad_heads) == nullptr);

    // A feature extractor whose resolved layer count disagrees with the
    // package's own conv_dim/conv_kernel/conv_stride lengths is a catalog
    // that resolved inconsistently.
    synth::omnivoice::ModelWeights short_feat = weights;
    short_feat.semantic_model.feat_conv.pop_back();
    SYNTH_TEST_CHECK(synth::omnivoice::build_semantic_branch(ctx, pcm, short_feat, hparams) == nullptr);

    // An unresolved positional-convolution bias.
    synth::omnivoice::ModelWeights unbound_pos = weights;
    unbound_pos.semantic_model.pos_conv.bias   = nullptr;
    SYNTH_TEST_CHECK(synth::omnivoice::build_semantic_branch(ctx, pcm, unbound_pos, hparams) == nullptr);

    // A SemanticEncoder res unit missing its (bias-free) second convolution.
    synth::omnivoice::ModelWeights unbound_res                       = weights;
    unbound_res.encoder_semantic.blocks[0].res_units[0].conv2.weight = nullptr;
    SYNTH_TEST_CHECK(synth::omnivoice::build_semantic_branch(ctx, pcm, unbound_res, hparams) == nullptr);
    return 0;
}

}  // namespace

int main() {
    SYNTH_TEST_CHECK(check_rejections() == 0);

    const size_t device_count = ggml_backend_dev_count();
    SYNTH_TEST_CHECK(device_count > 0);

    size_t exercised = 0;
    for (size_t index = 0; index < device_count; ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        std::printf("omnivoice-reference-encoder: %s (device type %d)\n", ggml_backend_dev_name(device), int(type));
        float max_diff = 0.0f;
        SYNTH_TEST_CHECK(run_case(device, max_diff));

        const float tolerance = type == GGML_BACKEND_DEVICE_TYPE_CPU ? kCpuTolerance : kAcceleratorTolerance;
        std::printf("  max_diff %.3g (tolerance %.3g)\n", double(max_diff), double(tolerance));
        SYNTH_TEST_CHECK(max_diff < tolerance);
        ++exercised;
    }
    SYNTH_TEST_CHECK(exercised > 0);
    return 0;
}

#include "catch.hpp"

// STL INCLUDES
#include<string>
#include<vector>
#include<list>
#include<complex>
#include <cstdlib>
// SYSTEM-INSTALLED LIBRARIES
#include <fftw3.h>
// LOCAL INCLUDES
#include "../include/helpers.hpp"
#include "../include/signal.hpp"
#include "../include/convolver.hpp"


////////////////////////////////////////////////////////////////////////////////
/// TESTING THE CONVOLVERPIPELINE CLASS
////////////////////////////////////////////////////////////////////////////////

// This test function returns the dot product between two signals, assuming
// that patch is smaller than sig. Offset is applied to patch, e.g
// dotprod += sig[i]*patch[i+offset]. If any index is out of bounds
float dotProdAt(FloatSignal &sig, FloatSignal &patch, const int offset){
  float result = 0;
  for(int i=0, j=offset, n=patch.getSize(), m=sig.getSize(); i<n; ++i, ++j){
    if(j>=0 && j<m){
      result += patch[i] * sig[j];
    }
  }
  return result;
}


TEST_CASE("test wisdom export/import", "[wisdom, fftw]"){
  MakeAndExportFftwWisdom("/tmp/test.wisdom", 0, 5, FFTW_PATIENT);
  ImportFftwWisdom("/tmp/test.wisdom");
}


TEST_CASE("Testing the OverlapSaveConvolver", "[OverlapSaveConvolver]"){
  FloatSignal signal([](long int x){return x+1;}, 10);
  FloatSignal patch1([](long int x){return 1+x;}, 6);
  OverlapSaveConvolver x(signal, patch1, true, true, 1);
  std::vector<FftTransformer*> sig_vec = x.getSignalVec();
  const size_t kExpectedPaddedSize = 16;
  const size_t kExpectedVectorSize = 3;
  const size_t kExpectedComplexSize = kExpectedPaddedSize/2+1;
  float sig_chunk1[kExpectedPaddedSize]{0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8};
  float sig_chunk2[kExpectedPaddedSize]{1,2,3,4,5,6,7,8,9,10,0,0,0,0,0,0};
  float sig_chunk3[kExpectedPaddedSize]{9,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  ComplexSignal cs_zeros(kExpectedComplexSize);

  // check that patch cannot be bigger than signal:
  REQUIRE_THROWS_AS(OverlapSaveConvolver(patch1, signal), std::runtime_error);

  SECTION("constructor, getPaddedSize, getPatch, getSignalVec"){
    // check padded patch size
    REQUIRE(x.getPaddedSize() == kExpectedPaddedSize);
    // check that padded patch contents were normalized
    float patch1_padded[kExpectedPaddedSize]{6,5,4,3,2,1,0,0,0,0,0,0,0,0,0,0};
    float* x_patch = x.getPatch()->r->begin();
    for(size_t i=0; i<kExpectedPaddedSize; ++i){
      REQUIRE(x_patch[i] == patch1_padded[i]/kExpectedPaddedSize);
    }
    // check chunked signal contents
    REQUIRE(sig_vec.size() == kExpectedVectorSize);
    for(size_t i=0; i<kExpectedPaddedSize; ++i){
      REQUIRE(sig_vec[0]->r->getData()[i] == sig_chunk1[i]);
      REQUIRE(sig_vec[1]->r->getData()[i] == sig_chunk2[i]);
      REQUIRE(sig_vec[2]->r->getData()[i] == sig_chunk3[i]);
    }
    // check that all complex signals are still zero:
    REQUIRE((cs_zeros == *(x.getPatch()->c)) == true);
    REQUIRE((cs_zeros == *(sig_vec[0]->c)) == true);
    REQUIRE((cs_zeros == *(sig_vec[1]->c)) == true);
    REQUIRE((cs_zeros == *(sig_vec[2]->c)) == true);
  }

  SECTION("test updatePatch"){
    // patch2 has different content and size, but same padded_size so it works
    FloatSignal patch2([](long int x){return 11+x;}, 4);
    x.updatePatch(patch2, true, false, false); // NO NORMALIZATION THIS TIME
    float patch2_pad[kExpectedPaddedSize]{14,13,12,11,0,0,0,0,0,0,0,0,0,0,0,0};
    float* x_patch = x.getPatch()->r->begin();
    for(size_t i=0; i<kExpectedPaddedSize; ++i){
      REQUIRE(x_patch[i] == patch2_pad[i]);
    }
    // test that c still zeros after the update
    REQUIRE((cs_zeros == *(x.getPatch()->c)) == true);
    // now update with fft_forward_after=true, and check that c has been set
    x.updatePatch(patch2, true, false, true);
    REQUIRE((cs_zeros == *(x.getPatch()->c)) == false);
    // test that a patch that is too long gets rejected and nothing else happens
    FloatSignal patch_too_long([](long int x){return 11+x;}, 20);
    REQUIRE_THROWS_AS(x.updatePatch(patch_too_long, true), std::runtime_error);
    for(size_t i=0; i<kExpectedPaddedSize; ++i){
      REQUIRE(x_patch[i] == patch2_pad[i]); // this still holds
    }
    REQUIRE((cs_zeros == *(x.getPatch()->c)) == false); // and this
  }


  SECTION("test updateSignal"){
    // first, check that different size throws an exception
    FloatSignal signal2_diff([](long int x){return x*(x%2)+1;}, 20);
    REQUIRE_THROWS_AS(x.updateSignal(signal2_diff, true), std::runtime_error);
    // and that no changes were made
    for(size_t i=0; i<kExpectedPaddedSize; ++i){
      REQUIRE(sig_vec[0]->r->getData()[i] == sig_chunk1[i]);
      REQUIRE(sig_vec[1]->r->getData()[i] == sig_chunk2[i]);
      REQUIRE(sig_vec[2]->r->getData()[i] == sig_chunk3[i]);
    }
    REQUIRE((cs_zeros == *(sig_vec[0]->c)) == true);
    REQUIRE((cs_zeros == *(sig_vec[1]->c)) == true);
    REQUIRE((cs_zeros == *(sig_vec[2]->c)) == true);

    // now for a valid new signal of same size, check that update is correct
    FloatSignal signal2_same([](long int x){return x*(x%2)+1;}, 10);
    x.updateSignal(signal2_same, false); // no forwarding!
    float sig2_chunk1[kExpectedPaddedSize]{0,0,0,0,0,0,0,0,1,2,1,4,1,6,1,8};
    float sig2_chunk2[kExpectedPaddedSize]{1,2,1,4,1,6,1,8,1,10,0,0,0,0,0,0};
    float sig2_chunk3[kExpectedPaddedSize]{1,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    REQUIRE(sig_vec.size() == kExpectedVectorSize);
    for(size_t i=0; i<kExpectedPaddedSize; ++i){
      REQUIRE(sig_vec[0]->r->getData()[i] == sig2_chunk1[i]);
      REQUIRE(sig_vec[1]->r->getData()[i] == sig2_chunk2[i]);
      REQUIRE(sig_vec[2]->r->getData()[i] == sig2_chunk3[i]);
    }
    // at this point, no forward FFT has beeen performed:
    REQUIRE((cs_zeros == *(sig_vec[0]->c)) == true);
    REQUIRE((cs_zeros == *(sig_vec[1]->c)) == true);
    REQUIRE((cs_zeros == *(sig_vec[2]->c)) == true);
    // but if we repeat it with the fft_after flag as true, changes are made:
    x.updateSignal(signal2_same, true);
    REQUIRE((cs_zeros == *(sig_vec[0]->c)) == false);
    REQUIRE((cs_zeros == *(sig_vec[1]->c)) == false);
    REQUIRE((cs_zeros == *(sig_vec[2]->c)) == false);
  }

    SECTION("test forwardPatch,backwardPatch,forwardSignal and backwardSignal"){
      // copy patch. FFT, normalize, then IFFT and compare result with copy
      FloatSignal padded_patch_cp(x.getPatch()->r->begin(),kExpectedPaddedSize);
      x.forwardPatch();
      *(x.getPatch()->c) *= 1.0/kExpectedPaddedSize;
      x.backwardPatch();
      for(size_t i=0; i<kExpectedPaddedSize; ++i){
        REQUIRE((x.getPatch()->r->getData()[i]-padded_patch_cp[i]) < 0.000001);
      }
      // same procedure with the signal chunks:
      FloatSignal sig_cp1(x.getSignalVec()[0]->r->begin(), kExpectedPaddedSize);
      FloatSignal sig_cp2(x.getSignalVec()[1]->r->begin(), kExpectedPaddedSize);
      FloatSignal sig_cp3(x.getSignalVec()[2]->r->begin(), kExpectedPaddedSize);
      x.forwardSignal();
      for(size_t i=0; i<kExpectedVectorSize; ++i){
        *(x.getSignalVec()[i]->c) *= 1.0/kExpectedPaddedSize;
      }
      x.backwardSignal();
      for(size_t i=0; i<kExpectedPaddedSize; ++i){
        REQUIRE((x.getSignalVec()[0]->r->getData()[i] - sig_cp1[i]) < 0.000001);
        REQUIRE((x.getSignalVec()[1]->r->getData()[i] - sig_cp2[i]) < 0.000001);
        REQUIRE((x.getSignalVec()[2]->r->getData()[i] - sig_cp3[i]) < 0.000001);
      }
    }

    SECTION("test spectralConv and extractConvolvedTo"){
      // declare signals and convolver
      for(size_t N=1; N<10; ++N){
        FloatSignal a([](long int x){return x+1;}, (100*N)-(N%10));
        FloatSignal b([](long int x){return x+1;}, N);
        OverlapSaveConvolver xxx(a, b, true, true, 1);
        // cross-correlation pipeline
        xxx.forwardPatch();
        xxx.forwardSignal();
        xxx.spectralConv();
        xxx.backwardSignal();
        // check results against dotprod
        size_t a_size = a.getSize();
        size_t b_size = b.getSize();
        size_t conv_size = a_size+b_size-1;
        FloatSignal cc_placeholder(conv_size);
        xxx.extractConvolvedTo(cc_placeholder);
        for(long int i=0; i<conv_size; ++i){
          REQUIRE(dotProdAt(a, b, i-(b_size-1)) == Approx(cc_placeholder[i]));
        }
      }
      // check accepted sizes by extractConvolvedTo: create signals
      const size_t kSigSz = 100;
      const size_t kPatchSz = 10;
      FloatSignal sig11(kSigSz);
      FloatSignal patch11(kPatchSz);
      OverlapSaveConvolver convolver11(sig11, patch11, true, true, 1);
      // create three containers with different sizes:
      FloatSignal xcorr_correct(kSigSz+kPatchSz-1);
      FloatSignal xcorr_small(kSigSz+kPatchSz-2);
      FloatSignal xcorr_big(kSigSz+kPatchSz+10);
      // test the containers: if they are big enough it won't crash
      convolver11.extractConvolvedTo(xcorr_correct);
      convolver11.extractConvolvedTo(xcorr_big, 5);
      // if the containers aren't big enough the convolver crashes
      REQUIRE_THROWS_AS(convolver11.extractConvolvedTo(xcorr_small),
                        std::runtime_error);
      REQUIRE_THROWS_AS(convolver11.extractConvolvedTo(xcorr_correct, 1),
                        std::runtime_error);
    }

    SECTION("Test cross-correlation using noise"){
      // basic settings
      size_t noise_size = 1000;
      size_t delayed_size = 2000;
      size_t delay = 456;
      // make noise signal and containers for the delayed and cross-correlated:
      FloatSignal noise([](long int x){return static_cast<float>(rand()) /
            static_cast<float>(RAND_MAX);}, noise_size);
      FloatSignal delayed(delayed_size);
      FloatSignal xcorr(noise_size+delayed_size-1);
      // create delayed
      noise *= 2;
      noise -= 1;
      delayed.addSignal(noise, delay);
      // calculate cross-correlation
      OverlapSaveConvolver xc(delayed, noise);
      xc.forwardPatch();
      xc.forwardSignal();
      xc.spectralConv();
      xc.backwardSignal();
      xc.extractConvolvedTo(xcorr);
      // calculate energy of noise signal and find absolute maximum of xcorr:
      float* noise_beg = noise.begin();
      float energy = std::inner_product(noise_beg, noise.end(), noise_beg, 0.0f);
      float* absmax = std::max_element(xcorr.begin(), xcorr.end(),
                                       abs_compare<float>);
      // require that maximum is at the delay point and equals the energy:
      REQUIRE(std::distance(xcorr.begin(), absmax) == noise_size+delay-1);
      REQUIRE(*absmax == Approx(energy));
      // require that the energy of the xcorr on the peak is > 99% of the total
      REQUIRE(*absmax * (*absmax)/energy > 0.99);
      // // test plots:
      // noise.plot("noise");
      // delayed.plot("delayed");
      // xcorr.plot("xcorr");
    }
}


// TEST ERROR FOR WRONG SIZES IN EXTRACT CONV.

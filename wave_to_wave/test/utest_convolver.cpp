#include "catch.hpp"

// STL INCLUDES
#include<string>
#include<vector>
#include<list>
#include<complex>
// SYSTEM-INSTALLED LIBRARIES
#include <fftw3.h>
// LOCAL INCLUDES
#include "../include/helpers.hpp"
#include "../include/signal.hpp"
#include "../include/convolver.hpp"


////////////////////////////////////////////////////////////////////////////////
/// TESTING THE CONVOLVER CLASS
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


TEST_CASE("Testing the OverlapSaveConvolver class", "[OverlapSaveConvolver]"){
  // create signals
  const size_t kSizeS = 50;
  const size_t kSizeP1 = 3;
  auto lin = [](long int n)->float {return n+1;};
  auto const1 = [](long int n)->float {return 1;};

  SECTION("test dotProdAt"){
    FloatSignal s(lin, kSizeS);
    FloatSignal p1(const1,  kSizeP1);
    REQUIRE(dotProdAt(s, p1, -1000) == 0);
    REQUIRE(dotProdAt(s, p1, -3) == 0);
    REQUIRE(dotProdAt(s, p1, -2) == 1);
    REQUIRE(dotProdAt(s, p1, -1) == 1+2);
    REQUIRE(dotProdAt(s, p1, 0) == 1+2+3);
    REQUIRE(dotProdAt(s, p1, 1) == 2+3+4);
    REQUIRE(dotProdAt(s, p1, 2) == 3+4+5);
    REQUIRE(dotProdAt(s, p1, kSizeS-1) == kSizeS);
    REQUIRE(dotProdAt(s, p1, kSizeS) == 0);
    REQUIRE(dotProdAt(s, p1, 1000) == 0);
  }

  SECTION("Convolver constructor and init fields"){
    FloatSignal s(lin, kSizeS);
    FloatSignal p1(lin, kSizeP1);
    FloatSignal p1_reversed(p1.getData(),p1.getSize());
    std::reverse(p1_reversed.begin(), p1_reversed.end());
    // instantiate convolver, and extract conv and xcorr
    OverlapSaveConvolver x(s, p1);
    x.executeConv();
    FloatSignal conv = x.extractResult();
    x.executeXcorr();
    FloatSignal xcorr = x.extractResult();
    // compare results with tests using dotProdAt
    for(size_t i=0, n=kSizeS+kSizeP1-1; i<n; ++i){
      REQUIRE(Approx(xcorr[i]) == dotProdAt(s, p1, i-kSizeP1+1));
      REQUIRE(Approx(conv[i]) == dotProdAt(s, p1_reversed, i-kSizeP1+1));
    }
  }
}

TEST_CASE("Testing the ConvolverPipeline", "[ConvolverPipeline]"){
  FloatSignal signal([](long int x){return x+1;}, 10);
  FloatSignal patch1([](long int x){return 1+x;}, 6);
  ConvolverPipeline x(signal, patch1, true, true, 1);
  std::vector<FftTransformer*> sig_vec = x.getSignalVec();
  const size_t kExpectedPaddedSize = 16;
  const size_t kExpectedVectorSize = 3;
  const size_t kExpectedComplexSize = kExpectedPaddedSize/2+1;
  float sig_chunk1[kExpectedPaddedSize]{0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8};
  float sig_chunk2[kExpectedPaddedSize]{1,2,3,4,5,6,7,8,9,10,0,0,0,0,0,0};
  float sig_chunk3[kExpectedPaddedSize]{9,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  ComplexSignal cs_zeros(kExpectedComplexSize);

  // check that patch cannot be bigger than signal:
  REQUIRE_THROWS_AS(ConvolverPipeline(patch1, signal), std::runtime_error);

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
    float patch2_padded[kExpectedPaddedSize]{14,13,12,11,0,0,0,0,0,0,0,0,0,0,0,0};
    float* x_patch = x.getPatch()->r->begin();
    for(size_t i=0; i<kExpectedPaddedSize; ++i){
      REQUIRE(x_patch[i] == patch2_padded[i]);
    }
    // test that c still zeros after the update
    REQUIRE((cs_zeros == *(x.getPatch()->c)) == true);
    // now update with fft_forward_after=true, and check that c has been set
    x.updatePatch(patch2, true, false, true);
    REQUIRE((cs_zeros == *(x.getPatch()->c)) == false);
    // test that a patch that is too long gets rejected and nothing else happens:
    FloatSignal patch_too_long([](long int x){return 11+x;}, 20);
    REQUIRE_THROWS_AS(x.updatePatch(patch_too_long, true), std::runtime_error);
    for(size_t i=0; i<kExpectedPaddedSize; ++i){
      REQUIRE(x_patch[i] == patch2_padded[i]); // this still holds
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


    SECTION("test multiplyPatchWithSig"){
      // declare signals and convolver
      FloatSignal a([](long int x){return x+1;}, 10);
      FloatSignal b([](long int x){return x+1;}, 3);
      ConvolverPipeline xxx(a, b, true, true, 1);
      // cross-correlation pipeline
      xxx.forwardPatch();
      xxx.forwardSignal();
      xxx.multiplyPatchWithSig();
      xxx.backwardSignal();
      // check results against dotprod
      const size_t padded_half=xxx.getPaddedSize()/2;
      auto vec = xxx.getSignalVec();
      long int a_size = a.getSize();
      long int b_size = b.getSize();
      for(long int i=0; i < a_size+b_size-1; ++i){
        long int vec_idx = (i+ padded_half)/padded_half - 1;
        long int in_vec_idx = i%padded_half + padded_half;
        REQUIRE(dotProdAt(a, b, i-(b_size-1)) ==
                Approx(vec[vec_idx]->r->begin()[in_vec_idx]));
      }
    }




      // for(size_t i=0; i<vec.size(); ++i){
      //   for(long int j=min_idx; j<max_idx; ++j){
      //     long int actual_idx = min_idx*i -min_idx + j - b.getSize() + 1;


      //     //REQUIRE(Approx(f->r->begin()[i]) ?? dotProdAt(a, b, ???));
      //   }
      // }


    // size_t downsampling = 50;
    // FloatSignal aaa([](long int x){return x%2 == 0;}, 44100*60/downsampling);
    // FloatSignal bbb([](long int x){return x%3 == 0;}, 44100*3/downsampling);
    // ConvolverPipeline xxx(aaa, bbb, 2048);
    // for(size_t i=0; i<1000*1000*1000; ++i){
    //   if(i%1000==0){std::cout << "i was " << i << std::endl;}
    //   xxx.updatePatch(bbb);
    //   xxx.forwardPatch();
    //   xxx.multiplyPatchWithSig();
    //   xxx.updateSignal(aaa);
    // }


}

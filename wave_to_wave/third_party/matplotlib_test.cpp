// https://github.com/lava/matplotlib-cpp
// g++ -Wall -Wextra matplotlib_test.cpp -std=c++11 -I/usr/include/python2.7 -lpython2.7 -o matplotlib_test && ./matplotlib_test

#include "matplotlibcpp.h"
#include <cmath>

namespace plt = matplotlibcpp;

void plot(const string name="FloatSignal", const size_t min_idx=0,
          const size_t max_idx=size_-1, const string style="r--",
          const string save_path="float_signal.png"){
  // TODO: CHECK THAT MIN>=0  AND MAX<=size_
  const size_t plot_range = 1+max_idx-min_idx;
  // make x and y axes. This is inefficient but plot is anyway much slower.
  // Better to have it here than to pollute the constructor with plot data.
  std::vector<float> domain(plot_range);
  std::vector<float> codomain(plot_range);
  for(int i=min_idx;i<=max_idx; ++i){
    domain[i]   = i;
    codomain[i] = data_[i];
  }
  plt::named_plot(name, domain, codomain, style);
  plt::xlim(min_idx, max_idx);
  plt::legend();
  if(!save_path.empty()){
    plt::save(save_path);
  }
  plt::show();
}

int main(){
  // Prepare data.
  int n = 5000;
  std::vector<double> x(n), y(n), z(n), w(n,2);
  for(int i=0; i<n; ++i) {
    x.at(i) = i*i;
    y.at(i) = sin(2*M_PI*i/360.0);
    z.at(i) = log(i);
  }

  // Plot line from given x and y data. Color is selected automatically.
  std::vector<double> ww(n); for(int i=0;i<n; ++i){ww[i]=i;}
  plt::named_plot("sin(x)", ww, y, "r--");
  // // Plot a red dashed line from given x and y data.
  // plt::plot(x, w,"r--");
  // // Plot a line whose name will show up as "log(x)" in the legend.
  // plt::named_plot("log(x)", x, z);

  // Set x-axis to interval [0,1000000]
  plt::xlim(0, n);//1000*1000);
  // Enable legend.
  plt::legend();
  // // Save the image (file format is determined by the extension)
  // plt::save("./basic.png");

  // show plots
  plt::show();



  // // Plot line from given x and y data. Color is selected automatically.
  // plt::plot(x, y);
  // // Plot a red dashed line from given x and y data.
  // plt::plot(x, w,"r--");
  // // Plot a line whose name will show up as "log(x)" in the legend.
  plt::named_plot("log(x)", x, z);

  // Set x-axis to interval [0,1000000]
  plt::xlim(0, 1000*1000);
  // Enable legend.
  plt::legend();
  // // Save the image (file format is determined by the extension)
  // plt::save("./basic.png");

  // show plots
  plt::show();

}

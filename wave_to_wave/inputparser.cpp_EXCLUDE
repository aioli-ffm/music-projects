// stl includes
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
// own header
#include "inputparser.h"
// namespace
using namespace std;


InputParser::InputParser (int &argc, char **argv)
  : originalPath(""), iterations(0), pickListPath(""), materialsPath(""),
    optListPath(""), action("optimize"), sampledownRatio(1), printFrequency(-1){
  string current_flag;
  for (int i=1; i < argc; ++i){
    string token = string(argv[i]);
    if (token[0]=='-'){
      current_flag = token;
      //cout << "flag changed to " << token << endl;
    } else{
      if (!current_flag.compare("-s")){
        originalPath = token;
      } else if (!current_flag.compare("-i")){
        iterations = stoi(token);
      } else if (!current_flag.compare("-p")){
        pickListPath = token;
      }  else if (!current_flag.compare("-m")){
        materialsPath = token;
      } else if (!current_flag.compare("-d")){
        optListPath = token;
      } else if (!current_flag.compare("-a")){
        action = token;
      } else if (!current_flag.compare("-r")){
        sampledownRatio = stoi(token);
      } else if (!current_flag.compare("-f")){
        printFrequency = stoi(token);
      } else {
        cout << "InputParser: malformed argument list " << current_flag << endl;
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// GETTERS
////////////////////////////////////////////////////////////////////////////////

string InputParser::getOriginalPath() const{
  return originalPath;
}

unsigned int InputParser::getIterations() const{
  return iterations;
}

string InputParser::getPickListPath() const{
  return pickListPath;
}

string InputParser::getMaterialsPath() const{
  return materialsPath;
}

string InputParser::getOptListPath() const{
  return optListPath;
}

string InputParser::getAction() const{
  return action;
}

unsigned int InputParser::getSampledownRatio() const{
  return sampledownRatio;
}

unsigned int InputParser::getPrintFrequency() const{
  return printFrequency;
}

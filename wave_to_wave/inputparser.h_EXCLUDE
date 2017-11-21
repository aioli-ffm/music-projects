#ifndef INPUTPARSER_H
#define INPUTPARSER_H

// stl includes

// namespace
using namespace std;

class InputParser{
 private:
  string originalPath;
  unsigned int iterations;
  string pickListPath;
  string materialsPath;
  string optListPath;
  string action;
  unsigned int sampledownRatio;
  int printFrequency;
 public:
  // constructor (parser-setter)
  InputParser(int &argc, char **argv);
  // deleted copy- and assignment constructors
  InputParser(const InputParser& copyFrom)=delete;
  InputParser& operator= (const InputParser& copyFrom)=delete;
  // getters
  string getOriginalPath() const;
  unsigned int getIterations() const;
  string getPickListPath() const;
  string getMaterialsPath() const;
  string getOptListPath() const;
  string getAction() const;
  unsigned int getSampledownRatio() const;
  unsigned int getPrintFrequency() const;
};


#endif

/**


    Copyright 2006 - 2010  Markus Peura, Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Rack.

    Rack is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Rack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU Lesser Public License
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

   Created on: Sep 30, 2010
*/
#include "Rack.h"
#include "RackLocalIf.h"
#include "../hi5/RaveConvert.h"

extern "C"
{
#include <rave_debug.h>
}

///////////////////////////
// prototypes
///////////////////////////

bool exec(string inputFileName, string outputFileName, std::vector<string>& args);
void writeToFile(RaveCoreObject* object, string outputFileName);
bool getFileNames(string& inputFile ,string& outputFile, std::vector<string>& args);
void getArgs(std::vector<string>& args, int argc, const char **argv);

///////////////////////////
// functions
///////////////////////////

/**
 * The main function called when executing the RACK binary from command line.
 * @param[in] argc - the number of arguments
 * @param[in] argv - the vector with arguments
 * @return Error code if something went wrong and 0 otherwise.
 */
int main(int argc, const char **argv) {
	bool passed = false;
	std::vector<string> args;
        string inputFileName = "";
	string outputFileName = "";

	Rave_initializeDebugger(); //Needed to initialize function ptr used when calling RAVE_ERROR1 etc.
	
	getArgs(args, argc, argv);
	passed = getFileNames(inputFileName , outputFileName, args);
	
	if (passed)
	{
		if (inputFileName == "") //No input file found. Probably starting with --help or similar
		{
			rack::Rack rack;
			passed = rack.main(NULL, args);
		}
		else
		{
			//Read input file and call RACK.
			cout << "Running RACK algorithms" << endl;
			passed = exec(inputFileName, outputFileName, args);
		}
	}

	return passed ? 0 : -1;
};

/**
 * Converts the standard char** argv to a vector of string for easier manipulation.
 * Removes also the first argument (the name of the binary).
 * @param[in/out] args - the vector to store all arguments in 
 * @param[in] argc - the number of arguments
 * @param[in] argv - the char** with arguments
 */
void getArgs(std::vector<string>& args, int argc, const char **argv)
{
  for (int i = 1; i < argc; i++) //skip program name (i.e. ./rack)
  {
    args.push_back(argv[i]);
  }
}

/**
 * Function which removes input and output file names from argument list. The file
 * names are stored in separate variables.
 * @param[in/out] inputFile - reference to variable for storing the input file name
 * @param[in/out] outputFile - reference to variable for storing the output file name
 * @param[in] args - the vector with arguments
 * @return True if execution passed, false otherwise.
 */
bool getFileNames(string& inputFileName, string& outputFileName, std::vector<string>& args)
{
	bool passed = true;
	const drain::RegExp h5FileExtension(".*\\.(h5|hdf5|hdf)$");

	//Get input file name
	if (args.size() >= 1 && h5FileExtension.test(args[0]))
	{
		//store file name and remove it from argv		
		inputFileName = args[0];
		args.erase(args.begin());
	}

	//Get output file name
	for (int i = 0; i < args.size(); i++)
	{
		if (args[i] == "--o")
		{
			if (i == args.size()-1 || !h5FileExtension.test(args[i+1])) //If last arg or incorrect file name
			{
				cerr << "Found --o without correct file name" << endl;
				passed = false;
			}
			else
			{
				outputFileName = args[i+1];				
				args.erase(args.begin()+i+1);
				args.erase(args.begin()+i);
				break; //only allow one output file per call
			}
		}
	}
	return passed;
}

/**
 * Writes a RaveCoreObject to file.ent list and stores them
 * @param[in] object - the object to write to file
 * @param[in] outputFileName - the file to write result to
 */
void writeToFile(RaveCoreObject* object, string outputFileName)
{
	cout << "Writing output to file " << outputFileName << endl;
	RaveIO_t* raveio = (RaveIO_t*)RAVE_OBJECT_NEW(&RaveIO_TYPE);
	RaveIO_setObject(raveio, object);
	int ok = RaveIO_save(raveio, outputFileName.c_str());
	RAVE_ASSERT((ok != 0), "Failed to save HDF5 file");
	RAVE_OBJECT_RELEASE(raveio);
}

/**
 * Intermediate function used when calling RACK from command line. The function reads
 * a HDF5 file an calls RACK with polar volume/scan and arguments.
 * @param[in] inputFileName - the file to read
 * @param[in] outputFileName - the file to write result to
 * @param[in] args - the vector with arguments
 * @return True if execution passed, false otherwise.
 */
bool exec(string inputFileName, string outputFileName, std::vector<string>& args)
{
	const char* filename = 	inputFileName.c_str();
	RaveIO_t* raveio = RaveIO_open(filename);
	Rave_ObjectType raveType = RaveIO_getObjectType(raveio);

	cout << "Found HDF5 file " << inputFileName << " of type " << raveType << endl;
	bool passed = true;

	if (raveType == Rave_ObjectType_PVOL) {
		
		PolarVolume_t* inputVolume = (PolarVolume_t*)RaveIO_getObject(raveio);

		//Call RACK and write results to file
		PolarVolume_t* outputVolume = runRack(inputVolume, args);
		if (outputVolume != NULL){
			if (outputFileName != "") {
				writeToFile((RaveCoreObject*)outputVolume, outputFileName);
			}
		} else {
			cerr << "RACK returned NULL ptr for PolarVolume" << endl;
			passed = false;
		}

		RAVE_OBJECT_RELEASE(inputVolume);
		RAVE_OBJECT_RELEASE(outputVolume);
	}
	else if (raveType == Rave_ObjectType_SCAN){
		PolarScan_t* inputScan = (PolarScan_t*)RaveIO_getObject(raveio);

		//Call RACK and write results to file
		PolarScan_t* outputScan = runRack(inputScan, args);

		if (outputScan != NULL){
			if (outputFileName != "") {
				writeToFile((RaveCoreObject*)outputScan, outputFileName);
			}
		} else {
			cerr << "RACK returned NULL ptr for PolarScan" << endl;
			passed = false;
		}
		RAVE_OBJECT_RELEASE(inputScan);
		RAVE_OBJECT_RELEASE(outputScan);
	}
	else{
		cout << "Input file is not a polar scan or polar volume. Type is " << raveType << ". Giving up ..." << endl;
		passed = false;
	}

	//cleanup
	RaveIO_close(raveio);
	RAVE_OBJECT_RELEASE(raveio);
	return passed;
}


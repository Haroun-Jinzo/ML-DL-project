#include "DataHandler.h"
#include <iostream>

DataHandler::DataHandler()
{
    /*dataArray = new std::vector<std::unique_ptr<data>>;
    testData = new std::vector<std::unique_ptr<data>>;
    trainingData = new std::vector<std::unique_ptr<data>>;
    validationData = new std::vector<std::unique_ptr<data>>;*/
}
DataHandler::~DataHandler()
{
    // Delete all data objects
    /*for (data* d : *dataArray) {
        delete d;
    }
    // Delete vector containers
    delete dataArray;
    delete testData;
    delete trainingData;
    delete validationData;*/
}

void DataHandler::Trim(std::string& s) {
    // Remove non-printable characters (including BOM remnants)
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) {
        return !std::isprint(static_cast<unsigned char>(c));
        }), s.end());

    // Trim whitespace
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
        }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
        }).base(), s.end());
}


// for simple dataset.csv
void DataHandler::read_csv(std::string path, std::string delimiter)  
{  
   numClasses = 0;  
   std::ifstream data_file;  
   data_file.open(path.c_str());  
   std::string line;  

   while (std::getline(data_file, line))  
   {  
       if (line.length() == 0) continue;  
       auto d = std::make_unique<data>();  // Use std::make_unique to create a unique_ptr  
       d->setNormalizedFeatureVector(std::make_unique<std::vector<double>>());  
       size_t position = 0;  
       std::string token;  
       while ((position = line.find(delimiter)) != std::string::npos)  
       {  
           token = line.substr(0, position);  
           d->append_to_feature_vector(std::stod(token));  
           line.erase(0, position + delimiter.length());  
       }  

       if (classFromString.find(line) != classFromString.end())  
       {  
           d->set_Label(classFromString[line]);  
       }  
       else  
       {  
           classFromString[line] = numClasses;  
           d->set_Label(classFromString[token]);  
           d->setNormalizedFeatureVector(std::make_unique<std::vector<double>>());  
           numClasses++;  
       }  
       dataArray.push_back(std::move(d));  // Use std::move to transfer ownership of the unique_ptr  
   }  
   for (auto& data : dataArray)  
       data->set_class_vector(numClasses);
   //normalize();  
   featureVectorSize = dataArray.front()->getNormalizedFeatureVector().size();  
}

// for dataset that requires encoding and stuff...
void DataHandler::read_csv(const std::string& path, const DataConfig& config, const std::string& delimiter) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
		throw std::runtime_error("Could not open file: " + path);
    }

    //Handle BOM
    if (file.peek() == 0xEF)
    {
        char bom[3];
        file.read(bom, 3);
        if (!(bom[1] == 0xBB && bom[2] == 0xBF))
        {
            file.seekg(0);
        }
    }

    std::string line;
    size_t lineNumber = 0;

    while (std::getline(file, line))
    {
        lineNumber++;
        try
        {
            processLine(line, config, delimiter, lineNumber);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("Error on Line " + std::to_string(lineNumber) + 
                "\nContent: " + line + "\nReason: " + e.what());
        }
    }

    if (dataArray.empty())
    {
        throw std::runtime_error("No valid data loaded");
    }

    //set class vectors for all samples
    for (auto& sample : dataArray)
    {
        sample->set_class_vector(numClasses);
    }
    featureVectorSize = dataArray.front()->getNormalizedFeatureVector().size();
}

void DataHandler::processLine(const std::string& line, const DataConfig& config, const std::string& delimiter, size_t lineNumber)
{
    std::istringstream ss(line);
    std::string token;
    auto sample = std::make_unique<data>();
    sample->setNormalizedFeatureVector(std::make_unique<std::vector<double>>());

    int colIndex = 0;
    std::string label;

    while (std::getline(ss, token, delimiter[0])) {
        Trim(token);

        if (colIndex == config.labelColumn) {
            label = token;
        }
        else {
            auto it = config.columnRules.find(colIndex);
            if (it == config.columnRules.end()) {
                throw std::runtime_error("Unexpected column index " +
                    std::to_string(colIndex));
            }

            switch (it->second) {
            case FeatureType::Numerical:
                sample->append_to_feature_vector(handleNumerical(token));
                break;
            case FeatureType::Categorical:
                sample->append_to_feature_vector(
                    handleCategorical(colIndex, token, config));
                break;
            case FeatureType::Ignore:
                break;
            }
        }
        colIndex++;
    }

    handleLabel(label, sample, config);
    dataArray.push_back(std::move(sample));
}

double DataHandler::handleNumerical(const std::string& token) {
    try {
        return std::stod(token);
    }
    catch (...) {
        throw std::runtime_error("Invalid numerical value: " + token);
    }
}

double DataHandler::handleCategorical(int colIndex,
    const std::string& token,
    const DataConfig& config) {
    // Check predefined encodings
    auto predef = config.categoricalEncodings.find(colIndex);
    if (predef != config.categoricalEncodings.end()) {
        auto it = predef->second.find(token);
        if (it != predef->second.end()) {
            return it->second;
        }
        if (!config.autoGenerateEncodings) {
            throw std::runtime_error("Unknown categorical value '" + token +
                "' in column " + std::to_string(colIndex));
        }
    }

    // Auto-generate encoding if allowed
    if (config.autoGenerateEncodings) {
        auto& encoding = categoricalMaps[colIndex];
        if (encoding.find(token) == encoding.end()) {
            encoding[token] = encoding.size();
        }
        return encoding[token];
    }

    throw std::runtime_error("Unhandled categorical value '" + token +
        "' in column " + std::to_string(colIndex));
}

void DataHandler::handleLabel(const std::string& token,
    std::unique_ptr<data>& sample,
    const DataConfig& config) {
    static std::unordered_map<std::string, int> labelMap;

    if (labelMap.find(token) == labelMap.end()) {
        if (config.autoGenerateEncodings) {
            labelMap[token] = numClasses++;
        }
        else {
            throw std::runtime_error("Unknown class label: " + token);
        }
    }

    sample->set_Label(labelMap[token]);
}

void DataHandler::readFeatureVector(const std::string& path)
{
    uint32_t header[4];
    unsigned char bytes[4];
    FILE* f = fopen(path.c_str(), "rb");
    if (f)
    {
        try {
            for (int i = 0; i < 4; i++)
            {
                if (fread(bytes, sizeof(bytes), 1, f) != 1) {
                    throw std::runtime_error("Failed to read file header");
                }
                header[i] = convertToLittleEndian(bytes);
            }

            printf("Done getting Input file header.\n");
            const int image_size = header[2] * header[3];

            for (int i = 0; i < header[1]; i++)
            {
                auto d = std::make_unique<data>();  // Use unique_ptr
                uint8_t element[1];

                for (int j = 0; j < image_size; j++)
                {
                    if (fread(element, sizeof(element), 1, f) != 1) {
                        throw std::runtime_error("Error reading image data at image " +
                            std::to_string(i) + " pixel " + std::to_string(j));
                    }
                    d->append_to_feature_vector(element[0]);
                }

                dataArray.push_back(std::move(d));  // Transfer ownership
            }

            printf("Successfully read and stored %zu feature vectors.\n", dataArray.size());
        }
        catch (const std::exception& e) {
            fclose(f);
            throw std::runtime_error(std::string("Error reading feature vectors: ") + e.what());
        }
        fclose(f);
    }
    else
    {
        throw std::runtime_error("Could not open file: " + path);
    }
}


void DataHandler::readFeatureLabels(std::string path)
{
    uint32_t header[2];
    unsigned char bytes[4];
    FILE* f = fopen(path.c_str(), "rb");
    if (f)
    {
        for (int i = 0; i < 2; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convertToLittleEndian(bytes);
            }
        }
        printf("Done getting lable file header.\n");
        for (int i = 0; i < header[1]; i++)
        {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f))
            {
                dataArray.at(i)->set_Label(element[0]);
            }
            else
            {
                printf("Error Reading From File.\n");
                exit(1);
            }
        }
        printf("successfully read and stored Labels.\n");
    }
    else
    {
        printf("could not read file");
        exit(1);
    }
}

void DataHandler::splitData()
{
    if (dataArray.empty()) {
        throw std::runtime_error("Cannot split empty dataset");
    }

    const double total = TrainSetPercent + TestSetPercent + ValidationPercent;
    if (std::abs(total - 1.0) > 0.001) {
        throw std::invalid_argument("Split percentages must sum to 1.0");
    }

    // Shuffle using move operations
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(dataArray.begin(), dataArray.end(), g);

    // Calculate sizes
    const size_t total_size = dataArray.size();
    const size_t train_size = static_cast<size_t>(total_size * TrainSetPercent);
    const size_t test_size = static_cast<size_t>(total_size * TestSetPercent);
    const size_t valid_size = total_size - train_size - test_size;

    // Clear previous splits
    trainingData.clear();
    testData.clear();
    validationData.clear();

    // Move elements using iterators
    auto start = std::make_move_iterator(dataArray.begin());
    auto mid_train = std::make_move_iterator(dataArray.begin() + train_size);
    auto mid_test = std::make_move_iterator(dataArray.begin() + train_size + test_size);
    auto end = std::make_move_iterator(dataArray.end());

    trainingData.insert(trainingData.end(), start, mid_train);
    testData.insert(testData.end(), mid_train, mid_test);
    validationData.insert(validationData.end(), mid_test, end);

    // Clear original data
    dataArray.clear();

    printf("Training Data Size: %zu\n", trainingData.size());
    printf("Test Data Size: %zu\n", testData.size());
    printf("Validation Data Size: %zu\n", validationData.size());
}


void DataHandler::countClasses()
{
	/*printf("Counting Classes...");
    int count = 0;
    for (unsigned i = 0; i < dataArray.size(); i++)
    {
        if (classFromInt.find(dataArray.at(i)->get_Label()) == classFromInt.end())
        {
            classFromInt[dataArray.at(i)->get_Label()] = count;
            dataArray.at(i)->setEnumeratedLabel(count);
            count++;
        }
    }
    numClasses = count;
    printf("Successfully Extracted %d Unique classes.\n", numClasses);*/

    std::unordered_set<int> uniqueLabels;
    for (const auto& sample : dataArray) {
        uniqueLabels.insert(sample->get_Label());
    }
    numClasses = uniqueLabels.size();

    if (numClasses < 2) {
        throw std::runtime_error("Need at least 2 distinct classes for classification");
    }

    std::cout << "Found " << numClasses << " classes in the dataset.\n";
}

void DataHandler::normalize()
{
    if (dataArray.empty()) return;

    // Calculate means and standard deviations
    size_t numFeatures = dataArray[0]->getNormalizedFeatureVector().size();
    std::vector<double> means(numFeatures, 0.0);
    std::vector<double> stddevs(numFeatures, 0.0);

    // Calculate means
    for (const auto& d : dataArray) {
        const auto& features = d->getNormalizedFeatureVector();
        for (size_t i = 0; i < numFeatures; ++i) {
            means[i] += features[i];
        }
    }
    for (auto& mean : means) mean /= dataArray.size();

    // Calculate standard deviations
    for (const auto& d : dataArray) {
        const auto& features = d->getNormalizedFeatureVector();
        for (size_t i = 0; i < numFeatures; ++i) {
            stddevs[i] += pow(features[i] - means[i], 2);
        }
    }
    for (auto& stddev : stddevs) stddev = sqrt(stddev / dataArray.size());

    // Apply z-score normalization
    for (auto& d : dataArray) {
        auto& features = d->getNormalizedFeatureVector();
        for (size_t i = 0; i < numFeatures; ++i) {
            if (stddevs[i] != 0) {
                features[i] = (features[i] - means[i]) / stddevs[i];
            }
        }
    }
}

uint32_t DataHandler::convertToLittleEndian(const unsigned char* bytes)
{
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}


int DataHandler::getClassCount()
{
    return numClasses;
}

std::vector<std::unique_ptr<data>>& DataHandler::getTrainingData()
{
    return trainingData;
}
std::vector<std::unique_ptr<data>>& DataHandler::getTestData()
{
    return testData;
}
std::vector<std::unique_ptr<data>>& DataHandler::getValidationData()
{
    return validationData;
}
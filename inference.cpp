#include "inference.h"

Inference::Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    cudaEnabled = runWithCuda;

    loadOnnxNetwork();
    // loadClassesFromFile(); The classes are hard-coded for this example
}


// 定义 reshape 函数将一维数组 reshape 为指定维度的二维数组
std::vector<std::vector<float>> reshape(const std::vector<float>& input, int rows, int cols) {
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = input[i * cols + j];
        }
    }

    return result;
}



// 定义 softmax 函数
std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> result(input.size(), std::vector<float>(input[0].size()));
    
    for (size_t i = 0; i < input.size(); ++i) {
        // 计算每行的 Softmax
        float sum_exp = 0.0;
        for (size_t j = 0; j < input[0].size(); ++j) {
            sum_exp += std::exp(input[i][j]);
            // std::cout << input[i][j] <<" -> exp: " <<  std::exp(input[i][j]) << std::endl;
        }
        // std::cout << "sum_exp: " <<  sum_exp << std::endl;   // inf ????
        // sum_exp += 0.000001f;
        for (size_t j = 0; j < input[0].size(); ++j) {
            result[i][j] = std::exp(input[i][j]) / sum_exp;
        }
    }
    return result;
}


std::vector<float> rowwise_multiply_and_sum(const std::vector<std::vector<float>>& data) {   //* 4x16

    std::vector<float> result(data.size(), 0.0);

    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            float memo = j;
            result[i] += (memo * data[i][j]);  // 乘以对应的元素
            // std::cout  << (data[i][j]) << " ";
        }
        // std::cout<< std::endl;
    }


    return result;
}


void valid_softmax_size(std::vector<std::vector<float>>& matrix)
{

    int rows = matrix.size(); // 4
    for (int i=0; i<rows; i++)
    {   
        std::cout<<i <<" : " << rows <<" * " << matrix[i].size() << std::endl;
    }
}


void valid_softmax_value(std::vector<std::vector<float>>& matrix)
{

    int rows = matrix.size(); // 4
    for (int i=0; i<rows; i++)
    {   
        for(int j=0; j<16; j++)
            std::cout<< matrix[i][j] << " ";
        std::cout << std::endl;
    }
}


void valid_softmax(std::vector<std::vector<float>>& matrix, int dim=1)
{

    int rows = matrix.size(); // 4
    int cols = matrix[0].size(); //16

    std::cout << "valid softmax:  \n";
    // int cou = 0;
    if (dim==1)
    {
        
        for (int i=0; i<cols; i++)
        {   
            float tmp=0;
            for(int j=0; j<rows; j++)
            {
                tmp += matrix[j][i];
                // cou ++;
                // std::cout <<cou<<" , " <<matrix[j][i] << std::endl;
            }
            std::cout << tmp <<" ";
        }
    }

    else
    {
        for (int i=0; i<rows; i++)
        {   
            float tmp=0;
            for(int j=0; j<16; j++)
            {
                tmp += matrix[i][j];
                std::cout  << matrix[i][j] << " ";
            }
            std::cout<< std::endl;

            std::cout << tmp <<" ";
            std::cout<< std::endl;
        }
    }
    




    std::cout<< std::endl;
    // exit(0);
        
}



std::vector<Detection> Inference::runInference(const cv::Mat &input)
{
    cv::Mat modelInput = input;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);

    std::cout <<"=> "<< modelInput.size()<< modelShape <<  std::endl;


    std::cout << "Blob Shape: [";
    for (int i = 0; i < blob.dims; ++i) {
        std::cout << blob.size[i];
        if (i < blob.dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // int rows = outputs[0].size[1];
    // int dimensions = outputs[0].size[2];

    // bool yolov8 = false;
    // // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])

    // if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    // {
    //     yolov8 = true;
    //     rows = outputs[0].size[2];
    //     dimensions = outputs[0].size[1];

    //     outputs[0] = outputs[0].reshape(1, dimensions);
    //     cv::transpose(outputs[0], outputs[0]);
    // }
    // float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / modelShape.width;
    float y_factor = modelInput.rows / modelShape.height;

    std::cout <<"modelInput cols: " << modelInput.cols  <<  std::endl;;
    std::cout <<"modelInput rows: " << modelInput.rows  << std::endl;;

    std::cout <<"modelShape cols: " << modelShape.width  << std::endl;;
    std::cout <<"modelShape rows: " << modelShape.height <<  std::endl;;


    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::cout <<"post process!!\n";
    std::cout <<"length of output = " << outputs.size() << std::endl;; // ==6
    int numHead = 3;

    // attention!!
    cv::Mat cls_8 = outputs[0];
    cv::Mat reg_8 = outputs[5];

    cv::Mat cls_16 = outputs[2];
    cv::Mat reg_16 = outputs[1];

    cv::Mat cls_32 = outputs[4];
    cv::Mat reg_32 = outputs[3];

    // int d = cls_8.size[1];
    // int h = cls_8.size[2];
    // int w = cls_8.size[3];

    // int d1 = reg_8.size[1];
    // int h1 = reg_8.size[2];
    // int w1 = reg_8.size[3];
    std::vector<cv::Mat> reconstrunct;
    reconstrunct.push_back(cls_8);
    reconstrunct.push_back(reg_8);

    reconstrunct.push_back(cls_16);
    reconstrunct.push_back(reg_16);

    reconstrunct.push_back(cls_32);
    reconstrunct.push_back(reg_32);

    std::vector<float> strides = {8,16,32};


    // std::cout << d <<" " << h <<" " << w << std::endl;;
    // std::cout << d1 <<" " << h1 <<" " << w1 << std::endl;;
    // exit(0);

    for(int index=0; index<numHead; index++)
    {   
        cv::Mat cls = reconstrunct[index * 2 + 0];
        cv::Mat reg = reconstrunct[index * 2 + 1]; //CV_32F 
        

        float* cls_ptr =  (float *)cls.data;
        float* reg_ptr =  (float *)reg.data;

        float stride = strides[index];

        int nc = cls.size[1];

        int h = reg.size[2];
        int w = reg.size[3];
        int h1 = cls.size[2];
        int w1 = cls.size[3];

        assert(w == w1 && h==h1); 

        int nd = reg.size[1];

        std::cout << h <<" " << w <<" " << nc << ",  " << nd << std::endl;;
        for (int c=0; c<nc; c++)
            for(int i=0; i <h; i++)
                for(int j=0; j<w; j++)
                {
                    float c_conf = *(cls_ptr + c*w*h + i*w + j);
                    if(c_conf > modelScoreThreshold)
                    {
                         confidences.push_back(c_conf);
                         class_ids.push_back(c);
                        // std::cout <<"conf: " << c_conf << std::endl;
                        std::vector<float> dist64(64, 0.0);  // 1*64*h*w
                        for(int k=0; k<nd; k++)  
                            dist64[k] = *(reg_ptr + k*w*h + i*w + j);

                        std::vector<std::vector<float>> dist4_16 = reshape(dist64, 4, 16); 
                        std::vector<std::vector<float>> dist4_16_sm = softmax(dist4_16);

                        // valid_softmax_value(dist4_16_sm);

                        std::vector<float> lt_rb = rowwise_multiply_and_sum(dist4_16_sm);

                        // for (int m=0; m<lt_rb.size(); m++)
                        // {
                        //     std::cout<< lt_rb[m] <<" ";
                        // }
                        float anchor_points0 = j + 0.5;
                        float anchor_points1 = i + 0.5;

                        float x1 = anchor_points0 - lt_rb[0];
                        float y1 = anchor_points1 - lt_rb[1];
                        float x2 = anchor_points0 + lt_rb[2];
                        float y2 = anchor_points1 + lt_rb[3];


                        float xmin = x1 * x_factor* strides[index];
                        float ymin = y1 * y_factor* strides[index];
                        float xmax = x2 * x_factor* strides[index];
                        float ymax = y2 * y_factor* strides[index];

                        xmin = xmin >0 ? xmin : 0;
                        ymin = ymin >0 ? ymin : 0;
                        xmax = xmax < modelInput.cols ? xmax : modelInput.cols;
                        ymax = ymax < modelInput.rows ? ymax : modelInput.rows;



                        // std::cout<< x1 <<" " << y1 <<" " <<x2 <<" " <<y2 <<" "<< std::endl;


                        int left = int(xmin);
                        int top = int(ymin);
                        int width = int(xmax - xmin);
                        int height = int(ymax-ymin);

                        boxes.push_back(cv::Rect(left, top, width, height));
                    }
                }     
    }


    std::vector<int> nms_result;
    std::cout<<"boxes num: " << boxes.size() << std::endl; //208

    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::cout<<"nms boxes num: " << nms_result.size() << std::endl; //22

    std::vector<Detection> detections{};


    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];
        //std::cout << idx <<" ";
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }
    //std::cout << std::endl;

    return detections;
}

void Inference::loadClassesFromFile()
{
    std::ifstream inputFile(classesPath);
    if (inputFile.is_open())
    {
        std::string classLine;
        while (std::getline(inputFile, classLine))
            classes.push_back(classLine);
        inputFile.close();
    }
}

void Inference::loadOnnxNetwork()
{
    net = cv::dnn::readNetFromONNX(modelPath);
    if (cudaEnabled)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat Inference::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

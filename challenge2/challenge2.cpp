#include <iostream>
#include <fstream>
#include <cstdio>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/IterativeSolvers>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

int main(int argc, char** argv)
{   
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    const char* input_image_path = argv[1];
 
    int width = 0, height = 0, channels = 0;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) {
        std::cerr << "Errore: impossibile caricare l'immagine " << input_image_path << "\n";
        return 1;
    }

    Eigen::MatrixXd A(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            A(i, j) = static_cast<double>(image_data[idx]); 
        }
    }

    //Task 1
    Eigen::MatrixXd AtA = A.transpose() * A;
    std::cout << "Euclidean norm: " << AtA.norm() << std::endl;

    stbi_image_free(image_data);

    //Task 2
    bool isSymmetricAtA = (AtA - Eigen::MatrixXd(AtA.transpose())).norm() < 1e-12;
    std::cout << "At*A è simmetrica? " << (isSymmetricAtA ? "Sì" : "No") << "\n";

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(AtA, Eigen::EigenvaluesOnly);
    if (eigensolver.info() != Eigen::Success) abort();
    Eigen::VectorXd lambda = eigensolver.eigenvalues(); // ordine crescente
    int n = lambda.size();
    std::cout << "Largest eigenvalue of A^T A: " << lambda(n - 1) << "\n";
    std::cout << "sigma_1 = " << std::sqrt(lambda(n - 1)) << "\n";
    std::cout << "sigma_2 = " << std::sqrt(lambda(n - 2)) << std::endl;

    //Task 3 e 4
    if (!Eigen::saveMarket(AtA, "AtA.mtx")) {
        std::cerr << "Errore: impossibile esportare AtA.mtx\n";
        return 1;
    }

    //Task 5
    Eigen::BDCSVD<Eigen::MatrixXd> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd W = svd.singularValues();
    std::cout << "Norm of W: " << W.norm() << std::endl;

    //Task 6
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    int k = 30;
    Eigen::MatrixXd C = U.leftCols(k);
    Eigen::MatrixXd D = V.leftCols(k) * W.head(k).asDiagonal();
 
    std::cout << "Numero di entry non nulle in C: " << C.nonZeros() << std::endl;
    std::cout << "Numero di entry non nulle in D: " << D.nonZeros() << std::endl;

    //Task 7
    


}
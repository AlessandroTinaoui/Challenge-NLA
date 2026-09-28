#include <Eigen/Sparse>
#include <vector>

// Build the mn x mn convolution matrix A such that  g = A * v
// reproduces  G = F (*) H  for an image F of size (rows x cols)
// stored ROW-MAJOR as a vector:  pixel (i,j)  ->  index  i*cols + j.
//
// H is the kernel, kh x kw (both odd so a true center exists).
// Challenge formula:  g_ij = sum_{k,l} f_{i+k-(kh-1)/2, j+l-(kw-1)/2} * H_kl
//
Eigen::SparseMatrix<double>
buildConvMatrix(int rows, int cols,
                const std::vector<std::vector<double>>& H)
{
    const int kh = H.size();        // kernel height (n in the PDF)
    const int kw = H[0].size();     // kernel width  (m in the PDF)
    const int ci = (kh - 1) / 2;    // center offset, rows  -> (n-1)/2
    const int cj = (kw - 1) / 2;    // center offset, cols  -> (m-1)/2

    const int N = rows * cols;      // system size: one eq per pixel
    Eigen::SparseMatrix<double> A(N, N);

    // WHY triplets: a convolution matrix is extremely sparse (<= kh*kw
    // nonzeros per row out of N columns). Filling a dense matrix would be
    // N*N memory. Triplet list -> setFromTriplets is the Eigen idiom for
    // assembling sparse matrices in one pass.
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(static_cast<size_t>(N) * kh * kw);  // upper bound, avoids reallocs

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            // Each OUTPUT pixel (i,j) is one ROW of A. The row says:
            // "output here = weighted sum of these input pixels".
            const int row = i * cols + j;

            for (int k = 0; k < kh; ++k) {
                for (int l = 0; l < kw; ++l) {

                    // Input pixel that weight H[k][l] multiplies.
                    // Shift by center so the kernel is centered on (i,j).
                    const int ii = i + k - ci;
                    const int jj = j + l - cj;

                    // WHY skip: "indices which make sense" in the PDF.
                    // Neighbors off the image edge -> term is zero, so we
                    // simply add no entry. This is zero-padding at borders.
                    if (ii < 0 || ii >= rows || jj < 0 || jj >= cols)
                        continue;

                    // COLUMN = index of that input pixel in vector v.
                    // A(row,col) = kernel weight. Multiplying A*v then
                    // gathers all neighbor contributions = the convolution.
                    const int col = ii * cols + jj;
                    trip.emplace_back(row, col, H[k][l]);

                    // NOTE: no duplicate (row,col) is ever produced because
                    // each (k,l) maps to a distinct neighbor. If a kernel
                    // COULD repeat a target, setFromTriplets SUMS duplicates,
                    // which is exactly the correct behavior anyway.
                }
            }
        }
    }

    A.setFromTriplets(trip.begin(), trip.end());
    return A;
}


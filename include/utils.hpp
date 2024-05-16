#ifndef HEADER_U
#   define HEADER_U
#include <fstream>
#include "boundary_conditions.hpp"

using namespace dealii;

double Hadamard(const SparseMatrix<double>& A, const SparseMatrix<double>& B);
void mat_to_vec(const MatrixType& mat, VectorType& vec);
void mat_to_vec(const MatrixType& mat, VectorTypeMPI& vec, IndexSet& locally_owned);
void vec_to_mat(const VectorTypeMPI& vec, MatrixType& mat, IndexSet& locally_owned);
void matMPI_to_mat(const MatrixTypeMPI& matMPI, SparseMatrix<double>& mat, IndexSet& locally_owned, int r_start, int r_end, int c_start);
void matMPI_to_mat(const SparseMatrixTypeMPI& matMPI, SparseMatrix<double>& mat, IndexSet& locally_owned, int r_start, int r_end);
void mat_to_matMPI(const SparseMatrix<double>& mat, SparseMatrixTypeMPI& matMPI, IndexSet& locally_owned, int r_start, int r_end, int c_start);
void vec_to_vecMPI(const BlockVector<double>& vec, VectorTypeMPI& vecMPI, IndexSet& locally_owned);
void element_copy(const SparseMatrix<double>& original, SparseMatrix<double>& copy );
void read_blockvector(BlockVector<double>& vec, std::string filename, int n_blocks);
void write_blockvector(BlockVector<double>& vec, std::string filename, int n_blocks);

double Hadamard(const SparseMatrix<double>& A, const SparseMatrix<double>& B) {
  double sum=0;
  for (unsigned int i=0;i<A.m();i++) {
    SparsityPattern::iterator row_begin = A.get_sparsity_pattern().begin(i);
    SparsityPattern::iterator row_end = A.get_sparsity_pattern().end(i);
    for (dealii::SparsityPattern::iterator it = row_begin; it != row_end; ++it)
      sum += A(i,it->column())*B(i,it->column());
  }
  return sum;
}

void mat_to_vec(const BlockSparseMatrix<double>& mat, BlockVector<double>& vec) {
  int dof_u = mat.block(0,0).m();
  int dof_p = mat.block(1,0).m();
  int N_PC = mat.n();
  for (unsigned int j=0;j<mat.n();j++)
    for (unsigned int i=0;i<mat.m();i++){
      int iq = i/dof_u;
      int ir = i%dof_u;
      vec[iq*(N_PC*dof_u + dof_p*j + ir - dof_u*j - i) + dof_u*j + i] = mat(i,j);
    }
}

void mat_to_vec(const MatrixType& mat, VectorTypeMPI& vec, IndexSet& locally_owned) {
  int dof_u = mat.block(0,0).m();
  int dof_p = mat.block(mat.n_block_rows()-1,mat.n_block_cols()-1).m();
  int N_PC = mat.n();
  for (const auto &i : locally_owned) {
    int iq = i/dof_u;
    int	ir = i%dof_u;
    for (int j=0;j<N_PC;j++)
      vec[iq*(N_PC*dof_u + dof_p*j + ir - dof_u*j - i) + dof_u*j + i] = mat(i,j);
  }
  vec.compress(VectorOperation::insert);
}

void vec_to_mat(const VectorTypeMPI& vec, MatrixType& mat, IndexSet& locally_owned) {
  int dof_u = mat.block(0,0).m();
  int dof_p = mat.block(mat.n_block_rows()-1,mat.n_block_cols()-1).m();
  int N_PC = mat.n();
  for (const auto &i : locally_owned) {
    int iq = i/dof_u;
    int	ir = i%dof_u;
    for (int j=0;j<N_PC;j++) 
      mat.set(i,j, vec[iq*(N_PC*dof_u + dof_p*j + ir - dof_u*j - i) + dof_u*j + i]);
  }
}

void matMPI_to_mat(const MatrixTypeMPI& matMPI, SparseMatrix<double>& mat, IndexSet& locally_owned, int r_start, int r_end, int c_start) {
  for (int i=r_start;i<r_end;i++) 
    if (locally_owned.is_element(i)) {
      SparsityPattern::iterator row_begin = mat.get_sparsity_pattern().begin(i-r_start);
      SparsityPattern::iterator row_end = mat.get_sparsity_pattern().end(i-r_start);
      for (dealii::SparsityPattern::iterator it = row_begin; it != row_end; ++it)
	mat.set(i-r_start,it->column(), matMPI(i,c_start + it->column()));
    }
}

void matMPI_to_mat(const SparseMatrixTypeMPI& matMPI, SparseMatrix<double>& mat, IndexSet& locally_owned, int r_start, int r_end) {
  for (int i=r_start;i<r_end;i++) 
    if (locally_owned.is_element(i)) {
      SparsityPattern::iterator row_begin = mat.get_sparsity_pattern().begin(i-r_start);
      SparsityPattern::iterator row_end = mat.get_sparsity_pattern().end(i-r_start);
      for (dealii::SparsityPattern::iterator it = row_begin; it != row_end; ++it) {
	mat.set(i-r_start,it->column(), matMPI(i-r_start, it->column()));
      }
    }
}

void mat_to_matMPI(const SparseMatrix<double>& mat, SparseMatrixTypeMPI& matMPI, IndexSet& locally_owned, int r_start, int r_end, int c_start) {
  for (int i=r_start;i<r_end;i++) 
    if (locally_owned.is_element(i)) {
      SparsityPattern::iterator row_begin = mat.get_sparsity_pattern().begin(i-r_start);
      SparsityPattern::iterator row_end = mat.get_sparsity_pattern().end(i-r_start);
      for (dealii::SparsityPattern::iterator it = row_begin; it != row_end; ++it)
	//std::cout<<mat(i-r_start, c_start + it->column())<<" "<<i<<" "<<it->column()<<" "<<matMPI.m()<<" "<<matMPI.n()<<std::endl;
	matMPI.set(i-r_start,c_start +it->column(), mat(i-r_start, it->column()));
    }
}

void vec_to_vecMPI(const BlockVector<double>& vec, VectorTypeMPI& vecMPI, IndexSet& locally_owned) {
  for (const auto &i : locally_owned) {
    vecMPI[i] = vec[i];
  }
  vecMPI.compress(VectorOperation::insert);
}

void element_copy(const SparseMatrix<double>& original, SparseMatrix<double>& copy ) {
  for (unsigned int i=0;i<original.m();i++){
    SparsityPattern::iterator row_begin = original.get_sparsity_pattern().begin(i);
    SparsityPattern::iterator row_end = original.get_sparsity_pattern().end(i);
    for (dealii::SparsityPattern::iterator it = row_begin; it != row_end; ++it)
      copy.set(i,it->column(), original(i,it->column()));
  }
}

void read_blockvector(BlockVector<double>& vec, std::string filename, int n_blocks) {
  for (int i=0;i<n_blocks;i++) {
    std::ifstream data_in(filename+std::to_string(i)+".txt");
    vec.block(i).block_read(data_in);
  }
}

void write_blockvector(BlockVector<double>& vec, std::string filename, int n_blocks) {
  for (int i=0;i<n_blocks;i++) {
    std::ofstream data_out(filename+std::to_string(i)+".txt");
    vec.block(i).block_write(data_out);
  }
}


#endif

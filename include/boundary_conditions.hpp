#ifndef HEADER_BC
#   define HEADER_BC
using namespace dealii;

#include <deal.II/lac/generic_linear_algebra.h>
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
}

typedef BlockVector<double> VectorType;
typedef BlockSparseMatrix<double> MatrixType;
typedef LA::MPI::BlockVector VectorTypeMPI;
typedef LA::MPI::BlockSparseMatrix MatrixTypeMPI;
typedef LA::MPI::SparseMatrix SparseMatrixTypeMPI;


class DirichletBoundary1 : public Function<2> {
  public:
    int N_PC;
  DirichletBoundary1(int N) : Function<2>(3*N), N_PC(N) {};
    virtual double value(const Point<2> & p, const unsigned int component = 0) const override;
    virtual void vector_value(const Point<2> &p, Vector<double> &  value) const override;
    virtual void vector_value(const Point<2> &p, VectorType &  value) const;
};

double DirichletBoundary1::value(const Point<2> & p, const unsigned int component) const {
  if (component == 0)
   return 20*(5-p[1])*(p[1]-2.5);
  else
   return 0;
  }

void DirichletBoundary1::vector_value(const Point<2> &p, Vector<double> &  values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = DirichletBoundary1::value(p, c);
}

void DirichletBoundary1::vector_value(const Point<2> &p, VectorType &  values) const {
  for (unsigned int c = 0; c < this->n_components; ++c) {
    values(c) = DirichletBoundary1::value(p, c);
  }
}

class DirichletBoundary3 : public Function<2> {
  public:
    int N_PC;
  DirichletBoundary3(int N) : Function<2>(3*N), N_PC(N){};
    virtual double value(const Point<2> & p, const unsigned int component = 0) const override;
    virtual void vector_value(const Point<2> &p, Vector<double> &  value) const override;
    virtual void vector_value(const Point<2> &p, VectorType &  value) const;
};

double DirichletBoundary3::value(const Point<2> & p, const unsigned int component) const {
 return 0;
  }

void DirichletBoundary3::vector_value(const Point<2> &p, Vector<double> &  values) const {
  for (unsigned int c = 0; c < this->n_components; ++c) 
    values(c) = DirichletBoundary3::value(p, c);
}
 
void DirichletBoundary3::vector_value(const Point<2> &p, VectorType &  values) const {
  for (unsigned int c = 0; c < this->n_components; ++c) {
    values(c) = DirichletBoundary3::value(p, c);
  }
}

#endif

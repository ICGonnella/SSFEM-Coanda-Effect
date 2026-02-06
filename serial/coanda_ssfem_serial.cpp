//#include <deal.II/base/quadrature_lib.h>
//#include <deal.II/base/function.h>
//#include <deal.II/base/utilities.h>
//#include <deal.II/base/tensor.h>
 
//#include <deal.II/lac/block_vector.h>
//#include <deal.II/lac/full_matrix.h>
//#include <deal.II/lac/block_sparse_matrix.h>
//#include <deal.II/lac/dynamic_sparsity_pattern.h>
//#include <deal.II/lac/solver_cg.h>
//#include <deal.II/lac/solver_gmres.h>
//#include <deal.II/lac/precondition.h>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/utilities.h>

 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
 
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
 
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/timer.h>

//#include <deal.II/numerics/error_estimator.h>
 
//#include <deal.II/numerics/solution_transfer.h>
 
//#include <deal.II/lac/sparse_direct.h>
 
//#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_snes.h>
#include <deal.II/numerics/nonlinear.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>

#include <fstream>
//#include <iostream>
 
using namespace dealii;

typedef BlockVector<double> VectorType;
typedef BlockSparseMatrix<double> MatrixType;
//typedef LA::MPI::BlockVector VectorTypeMPI;
//typedef LA::MPI::BlockSparseMatrix MatrixTypeMPI;

void read_csv_matrix(const std::string &filename,FullMatrix<double> &M){
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Impossibile aprire il file " + filename);

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)){
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ','))
            row.push_back(std::stod(cell));
        data.push_back(row);
    }

    if (data.empty())
        throw std::runtime_error("File CSV vuoto");

    const unsigned int n_rows = data.size();
    const unsigned int n_cols = data[0].size();

    for (const auto &row : data)
        if (row.size() != n_cols)
            throw std::runtime_error("CSV non rettangolare");
    M.reinit(n_rows, n_cols);
    for (unsigned int i = 0; i < n_rows; ++i)
        for (unsigned int j = 0; j < n_cols; ++j)
            M(i, j) = data[i][j];
}

void read_csv_vector(const std::string &filename, Vector<double> &v){
  std::ifstream file(filename);
  if (!file)
    throw std::runtime_error("Impossibile aprire il file " + filename);

  std::vector<std::vector<double>> data;
  std::string line;

  while (std::getline(file, line))
  {
    if (line.empty())
      continue;

    std::stringstream ss(line);
    std::string cell;
    std::vector<double> row;

    while (std::getline(ss, cell, ','))
    {
      if (!cell.empty())
        row.push_back(std::stod(cell));
    }

    if (!row.empty())
      data.push_back(row);
  }

  if (data.empty())
    throw std::runtime_error("File CSV vuoto: " + filename);

  const unsigned int n_rows = data.size();
  const unsigned int n_cols = data[0].size();

  for (const auto &row : data)
    if (row.size() != n_cols)
      throw std::runtime_error("CSV non rettangolare: " + filename);

  // Caso 1: vettore riga (1 x N)
  if (n_rows == 1)
  {
    v.reinit(n_cols);
    for (unsigned int j = 0; j < n_cols; ++j)
      v[j] = data[0][j];
  }
  // Caso 2: vettore colonna (N x 1)
  else if (n_cols == 1)
  {
    v.reinit(n_rows);
    for (unsigned int i = 0; i < n_rows; ++i)
      v[i] = data[i][0];
  }
  else
  {
    throw std::runtime_error(
      "Il CSV non rappresenta un vettore (né riga né colonna): " + filename);
  }
}

//----------------------------------------DB1---------------------------------------------
class DirichletBoundary1 : public Function<2> {
  public:
    explicit DirichletBoundary1(const unsigned int npc): Function<2>(3*npc), npc(npc) {};
    virtual double value(const Point<2> & p, const unsigned int component = 0) const override;
    virtual void vector_value(const Point<2> &p, Vector<double> &  value) const override;
  private:
    const unsigned int npc;
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

//----------------------------------------DB3---------------------------------------------
class DirichletBoundary3 : public Function<2> {
  public:
    explicit DirichletBoundary3(const unsigned int npc): Function<2>(3*npc), npc(npc) {};
    virtual double value(const Point<2> & p, const unsigned int component = 0) const override;
    virtual void vector_value(const Point<2> &p, Vector<double> &  value) const override;
  private:
    const unsigned int npc;
};

double DirichletBoundary3::value(const Point<2> & p, const unsigned int component) const {
 return 0;
  }
 
void DirichletBoundary3::vector_value(const Point<2> &p, Vector<double> &  values) const {
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = DirichletBoundary3::value(p, c);
}

class StationaryCoanda : public ParameterAcceptor
{
public:
  StationaryCoanda();
  void run(std::string input_file);
  VectorType solution;
  MPI_Comm mpi_communicator;

private:
  void make_grid();
  void setup_system();
  void upload_grid();
  void initialize(std::string input_file);
  void compute_residual(const VectorType &X, VectorType &R);
  void compute_time_dep_Jacobian(const VectorType &X);
  void solve_system();
  void output_results();

  //---------------TIMER------------------------
  dealii::TimerOutput timer;
  std::ofstream timer_file;
 
  //---------------TO INITIALIZE---------------------
  int fe_degree;
  int n_glob_ref;
  std::string mesh_input_file;
  std::string output_file;

  //--------------PRECONDITIONERS--------------------
  SparseILU<double> A_prec;
  PreconditionSSOR<SparseMatrix<double>> Mp_prec;
  SparseMatrix<double> Mp;
  SparsityPattern sp_p;
  types::global_dof_index pressure_offset;

  //------------------GRID------------------------
  //parallel::distributed::Triangulation<2> triangulation;

 
  std::vector<types::global_dof_index> dofs_per_block;
  std::vector<types::global_dof_index> dofs_per_block_s;
  std::vector<types::global_dof_index> dofs_per_block_init;

  Triangulation<2> triangulation;
  std::unique_ptr<FESystem<2>> fe;
  DoFHandler<2>    dof_handler;
  std::unique_ptr<FESystem<2>> fe_s;
  DoFHandler<2>    dof_handler_s;
  std::unique_ptr<FESystem<2>> fe_init;
  DoFHandler<2>    dof_handler_init;


  std::string rhs_string;
  std::map<std::string, double> constants;
  std::unique_ptr<FunctionParser<2>> rhs;
  
  AffineConstraints<double> constraints;
  AffineConstraints<double> zero_constraints;
  
  BlockSparsityPattern sparsity_pattern;
  BlockSparsityPattern sparsity_pattern_s;
  MatrixType system_matrix_s;
  MatrixType static_system_matrix_s;
  std::vector<FullMatrix<double>> stochastic_matrix;
  Vector<double> norms;

  int n_stochastic_matrix;
  int npc;
  VectorType current_solution; 
  VectorType system_rhs_s;
  //PETScWrappers::PreconditionLU system_matrix_factorization;
  NonlinearSolverSelector<VectorType>::AdditionalData::SolverType solver_type;
  NonlinearSolverSelector<VectorType>::AdditionalData::SolutionStrategy strategy;
  std::string solver_type_string;
  std::string strategy_string;
  int max_iter;
  double function_tolerance;
  double init_solution;
  int n_blocks_load;
  int npc_init;
  
  //SparseMatrix<double>      pressure_mass_matrix;

  //BlockVector<double> present_solution;
  //BlockVector<double> newton_update;
  //BlockVector<double> evaluation_point;
};

StationaryCoanda::StationaryCoanda()
  : ParameterAcceptor("NavierStokes")
  , dof_handler(triangulation)
  , dof_handler_s(triangulation)
  , dof_handler_init(triangulation)
  , timer_file("timing.txt", std::ios::app)
  , timer(timer_file, dealii::TimerOutput::summary, dealii::TimerOutput::wall_times)
    //, mpi_communicator(MPI_COMM_WORLD)
    //, n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    //, this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    //, pcout(std::cout, (this_mpi_process == 0))
{
  add_parameter("Finite element degree", fe_degree);
  add_parameter("Number of global refinements", n_glob_ref);
  add_parameter("Force term", rhs_string);
  add_parameter("Output file", output_file);
  add_parameter("Constants", constants);
  add_parameter("Mesh file", mesh_input_file);
  add_parameter("Solver Type", solver_type_string);
  add_parameter("Solution Strategy", strategy_string);
  add_parameter("Max iteration number", max_iter);
  add_parameter("Nonlinear solver tolerance", function_tolerance);
  add_parameter("Number of stochastic matrices", n_stochastic_matrix);
  add_parameter("NPC", npc);
  add_parameter("Init solution", init_solution);
  add_parameter("Number of init blocks to load", n_blocks_load);
  add_parameter("npc of the saved solution", npc_init);
}

void StationaryCoanda::initialize(std::string input_file){
  ParameterAcceptor::initialize(input_file);
  fe = std::make_unique<FESystem<2>>(FE_Q<2>(fe_degree + 1), 2, FE_Q<2>(fe_degree), 1);
  fe_s = std::make_unique<FESystem<2>>(FE_Q<2>(fe_degree + 1), 2*npc, FE_Q<2>(fe_degree), 1*npc);
  fe_init = std::make_unique<FESystem<2>>(FE_Q<2>(fe_degree + 1), 2*npc_init, FE_Q<2>(fe_degree), 1*npc_init);
  rhs = std::make_unique<FunctionParser<2>>(1);
  rhs->initialize(rhs->default_variable_names(), rhs_string, constants);
  if (solver_type_string=="automatic") 
    solver_type = NonlinearSolverSelector<VectorType>::AdditionalData::SolverType::automatic;
  if (solver_type_string=="nox") 
    solver_type = NonlinearSolverSelector<VectorType>::AdditionalData::SolverType::nox;
  if (solver_type_string=="petsc_snes")
      solver_type = NonlinearSolverSelector<VectorType>::AdditionalData::SolverType::petsc_snes;
  if (solver_type_string=="kinsol")
      solver_type = NonlinearSolverSelector<VectorType>::AdditionalData::SolverType::kinsol;
  if (strategy_string=="newton")
      strategy = NonlinearSolverSelector<VectorType>::AdditionalData::SolutionStrategy::newton;
  if (strategy_string=="linesearch")
      strategy = NonlinearSolverSelector<VectorType>::AdditionalData::SolutionStrategy::linesearch;
  if (strategy_string=="picard")
      strategy = NonlinearSolverSelector<VectorType>::AdditionalData::SolutionStrategy::picard;
}
void StationaryCoanda::upload_grid(){
  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream mesh_file(mesh_input_file);
  grid_in.read_msh(mesh_file);
  mesh_file.close();

  for (const auto &face : triangulation.active_face_iterators()) {
    if (face->at_boundary()) {
      if (std::fabs(face->center()[0]) < 1e-12)             // Inlet boundary
	face->set_boundary_id(1);
      else if (std::fabs(face->center()[0] - 50.0) < 1e-12) // Outer boundary
	face->set_boundary_id(2);
      else                                                  // Wall boundary
	face->set_boundary_id(3);
    }
  }
  
  std::ofstream out("mesh.vtk");
  GridOut grid_out;
  grid_out.write_vtk(triangulation, out);
}
void StationaryCoanda::make_grid(){
  Triangulation<2> rectangle;
  GridGenerator::hyper_rectangle(rectangle, Point<2>(0,0), Point<2>(50,7.5));
  rectangle.refine_global(n_glob_ref);
  std::set<typename Triangulation<2>::active_cell_iterator> cells_to_remove;
  bool inside_domain = true;
  for (const auto &cell : rectangle.active_cell_iterators()) {
    inside_domain = true;
    for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v) 
      if ((cell->vertex(v)[0]<10) && ((cell->vertex(v)[1]>5 && cell->vertex(v)[1]<7.5) || (cell->vertex(v)[1]<2.5))) 
	inside_domain = false;
    if (!inside_domain)
      cells_to_remove.insert(cell);
  }
  GridGenerator::create_triangulation_with_removed_cells(rectangle, cells_to_remove, triangulation);
  for (const auto &face : triangulation.active_face_iterators()) {
    if (face->at_boundary()) {
      if (std::fabs(face->center()[0]) < 1e-12) {             // Inlet boundary
	face->set_boundary_id(1);
      }
      else {
	if (std::fabs(face->center()[0] - 50.0) < 1e-12) // Outer boundary
	  face->set_boundary_id(2);
	else                                                  // Wall boundary
	  face->set_boundary_id(3);
      }
    }
  }
  
  std::ofstream out("mesh.vtk");
  GridOut grid_out;
  grid_out.write_vtk(triangulation, out);
}
void StationaryCoanda::setup_system(){
  //GridTools::partition_triangulation(n_mpi_processes, triangulation);
  dof_handler.distribute_dofs(*fe);
  std::vector<unsigned int> block_component(3,0);
  block_component[2] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);
  const FEValuesExtractors::Vector velocities(0);

  dof_handler_s.distribute_dofs(*fe_s);
  std::vector<unsigned int> block_component_s(3*npc,0);
  std::vector<FEValuesExtractors::Vector> velocities_s;
  velocities_s.reserve(npc);
  for (unsigned int i = 0; i < npc; ++i) {
    block_component_s[i*2] = i;
    block_component_s[i*2+1] = i;
    block_component_s[npc*2+i] = i+npc;
    velocities_s.emplace_back(2*i);
  }
  DoFRenumbering::component_wise(dof_handler_s, block_component_s);

  dof_handler_init.distribute_dofs(*fe_init);
  std::vector<unsigned int> block_component_init(3*npc_init,0);
  for (unsigned int i = 0; i < npc_init; ++i) {
    block_component_init[i*2] = i;
    block_component_init[i*2+1] = i;
    block_component_init[npc_init*2+i] = i+npc_init;
  }
  DoFRenumbering::component_wise(dof_handler_init, block_component_init);

  
  dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  unsigned int dof_u = dofs_per_block[0];
  unsigned int dof_p = dofs_per_block[1];

  dofs_per_block_s = DoFTools::count_dofs_per_fe_block(dof_handler_s, block_component_s);
  dofs_per_block_init = DoFTools::count_dofs_per_fe_block(dof_handler_init, block_component_init);

  std::cout << "Number of active cells: "<< triangulation.n_active_cells()<< std::endl << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("<<dof_u << " + "<< dof_p <<")"<<std::endl;

  {
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler_s, constraints);
  for (int j=0;j<npc;j++) {
    VectorTools::interpolate_boundary_values(dof_handler_s, 1, DirichletBoundary1(npc), constraints, fe_s->component_mask(velocities_s[j]));
    VectorTools::interpolate_boundary_values(dof_handler_s, 3, DirichletBoundary3(npc), constraints, fe_s->component_mask(velocities_s[j]));
  }
  }
  constraints.close();

  {
  zero_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler_s, zero_constraints);
  for (int j=0;j<npc;j++) {
    VectorTools::interpolate_boundary_values(dof_handler_s, 1, DirichletBoundary3(npc), zero_constraints, fe_s->component_mask(velocities_s[j]));
    VectorTools::interpolate_boundary_values(dof_handler_s, 3, DirichletBoundary3(npc), zero_constraints, fe_s->component_mask(velocities_s[j]));
  }
  }
  zero_constraints.close();

  BlockDynamicSparsityPattern dsp_s(dofs_per_block_s, dofs_per_block_s);
  DoFTools::make_sparsity_pattern(dof_handler_s, dsp_s, constraints,false);

  sparsity_pattern_s.copy_from(dsp_s);
  system_matrix_s.reinit(sparsity_pattern_s);
  static_system_matrix_s.reinit(sparsity_pattern_s);


  solution.reinit(dofs_per_block_s);
  current_solution.reinit(dofs_per_block_s);
  system_rhs_s.reinit(dofs_per_block_s);

  stochastic_matrix.resize(npc+2);
  for (int i=0;i<npc+2;i++) {
    stochastic_matrix[i].reinit(npc,npc);
    read_csv_matrix("build/M_" + std::to_string(i) + ".csv",stochastic_matrix[i]);
  }
  norms.reinit(npc);
  read_csv_vector("build/norms.csv",norms);
}
void StationaryCoanda::compute_residual(const VectorType &X, VectorType &R){
  R = 0.;
  QGauss<2> quadrature(fe_degree + 2);
  FEValues<2> fe_values(*fe_s,
                        quadrature,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe_s->n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature.size();

  // Extractor for fe_s:
  std::vector<FEValuesExtractors::Vector> vel(npc, FEValuesExtractors::Vector(0));
  std::vector<FEValuesExtractors::Scalar> pres(npc, FEValuesExtractors::Scalar(0));
  for (unsigned int j = 0; j < npc; ++j) {
    vel[j]  = FEValuesExtractors::Vector(2*j);
    pres[j] = FEValuesExtractors::Scalar(2*npc + j);
  }

  // Storage for solution values for each mode s
  std::vector<std::vector<Tensor<1,2>>> u_mode(npc, std::vector<Tensor<1,2>>(n_q_points));
  std::vector<std::vector<Tensor<2,2>>> grad_u_mode(npc, std::vector<Tensor<2,2>>(n_q_points));
  std::vector<std::vector<double>>      div_u_mode(npc, std::vector<double>(n_q_points));
  std::vector<std::vector<double>>      p_mode(npc, std::vector<double>(n_q_points));

  std::vector<std::vector<Tensor<2,2>>> grad_u_M0(npc,  std::vector<Tensor<2,2>>(n_q_points));
  std::vector<std::vector<Tensor<1,2>>> u_grad_u_M2(npc,std::vector<Tensor<1,2>>(n_q_points));
  std::vector<std::vector<double>>      p_M1(npc,    std::vector<double>(n_q_points));
  std::vector<std::vector<double>>      div_u_M1(npc,  std::vector<double>(n_q_points));

  Vector<double> local_vector(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler_s.active_cell_iterators()){
    fe_values.reinit(cell);
    local_vector = 0.;

    for (unsigned int s = 0; s < npc; ++s){
      fe_values[vel[s]].get_function_values(X, u_mode[s]);
      fe_values[vel[s]].get_function_gradients(X, grad_u_mode[s]);
      fe_values[vel[s]].get_function_divergences(X, div_u_mode[s]);
      fe_values[pres[s]].get_function_values(X, p_mode[s]);
    }

    // coupled terms with stochastic matrices
    for (unsigned int j = 0; j < npc; ++j){
      for (unsigned int q = 0; q < n_q_points; ++q){
        grad_u_M0[j][q]   = Tensor<2,2>();
        u_grad_u_M2[j][q] = Tensor<1,2>();
        p_M1[j][q]        = 0.0;
        div_u_M1[j][q]      = 0.0;
        for (unsigned int s = 0; s < npc; ++s){
          grad_u_M0[j][q]   += stochastic_matrix[0](s, j)   * grad_u_mode[s][q];
          p_M1[j][q]        += stochastic_matrix[1](s, j)   * p_mode[s][q];
          div_u_M1[j][q]      += stochastic_matrix[1](s, j)   * div_u_mode[s][q];
          for (unsigned int r = 0; r < npc; ++r){
            u_grad_u_M2[j][q] += stochastic_matrix[2 + j](s, r) * (grad_u_mode[s][q] * u_mode[r][q]);
          }
        }
      }
    }

    // assemble local vector
    for (unsigned int i = 0; i < dofs_per_cell; ++i){
      const unsigned int comp = fe_s->system_to_component_index(i).first;

      if (comp < 2 * npc){
        const unsigned int j = comp / 2;
        
        for (unsigned int q = 0; q < n_q_points; ++q){
          const Tensor<1,2> phi_u_i = fe_values[vel[j]].value(i, q);
          const Tensor<2,2> grad_phi_u_i = fe_values[vel[j]].gradient(i, q);
          const double div_phi_u_i  = fe_values[vel[j]].divergence(i, q);

          const double visc = scalar_product(grad_phi_u_i, grad_u_M0[j][q]);
          const double pres_term = -(div_phi_u_i * p_M1[j][q]);
          const double conv = u_grad_u_M2[j][q] * phi_u_i;

          local_vector(i) += (visc + pres_term + conv) * fe_values.JxW(q);
        }
      }
      else{
        const unsigned int j = comp - 2 * npc;

        for (unsigned int q = 0; q < n_q_points; ++q){
          const double phi_p_i = fe_values[pres[j]].value(i, q);

          local_vector(i) += ( - phi_p_i * div_u_M1[j][q] ) * fe_values.JxW(q);
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    zero_constraints.distribute_local_to_global(local_vector, local_dof_indices, R);
  }
}
void StationaryCoanda::compute_time_dep_Jacobian(const VectorType &X) {
  system_matrix_s=0;
  system_rhs_s = 0;

  QGauss<2> quadrature_formula(fe_degree + 2);
  FEValues<2> fe_values(*fe_s,
                        quadrature_formula,
                        update_values | update_quadrature_points | update_JxW_values | update_gradients);
  
  const unsigned int dofs_per_cell = fe_s->n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();
  // Extractor for fe_s:
  std::vector<FEValuesExtractors::Vector> vel(npc, FEValuesExtractors::Vector(0));
  std::vector<FEValuesExtractors::Scalar> pres(npc, FEValuesExtractors::Scalar(0));
  for (unsigned int j = 0; j < npc; ++j) {
    vel[j]  = FEValuesExtractors::Vector(2*j);
    pres[j] = FEValuesExtractors::Scalar(2*npc + j);
  }

  std::vector<std::vector<Tensor<1,2>>> u_mode(npc, std::vector<Tensor<1,2>>(n_q_points));
  std::vector<std::vector<Tensor<2,2>>> grad_u_mode(npc, std::vector<Tensor<2,2>>(n_q_points));

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler_s.active_cell_iterators()) {
    fe_values.reinit(cell);
    local_matrix = 0;
    local_rhs    = 0;

    for (unsigned int j = 0; j < npc; ++j){
      fe_values[vel[j]].get_function_values(X, u_mode[j]);
      fe_values[vel[j]].get_function_gradients(X, grad_u_mode[j]);
    }

    for (unsigned int q = 0; q < n_q_points; ++q) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int comp_i = fe_s->system_to_component_index(i).first;
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          const unsigned int comp_j = fe_s->system_to_component_index(j).first;
          double int_arg = 0.0;
          double int_arg_time_dep = 0.0;
          if (comp_i < 2 * npc){
            const unsigned int j_i = comp_i / 2;
            const Tensor<2,2> grad_phi_u_i = fe_values[vel[j_i]].gradient(i, q);
            const Tensor<1,2> phi_u_i = fe_values[vel[j_i]].value(i, q);
            const double div_phi_u_i  = fe_values[vel[j_i]].divergence(i, q);
            if (comp_j < 2 * npc){
              const unsigned int j_j = comp_j / 2;
              const Tensor<2,2> grad_phi_u_j = fe_values[vel[j_j]].gradient(j, q);
              const double div_phi_u_j  = fe_values[vel[j_j]].divergence(j, q);
              const Tensor<1,2> phi_u_j = fe_values[vel[j_j]].value(j, q);
              int_arg += scalar_product(grad_phi_u_i, grad_phi_u_j)*stochastic_matrix[0](j_i,j_j);
              for (unsigned int l = 0; l < npc; ++l){
                int_arg_time_dep += (grad_phi_u_j*u_mode[l][q]+grad_u_mode[l][q]*phi_u_j)*phi_u_i*stochastic_matrix[2+l](j_i,j_j);
              }
            }
            else{
              const unsigned int j_j = comp_j - 2*npc;
              const double phi_p_j = fe_values[pres[j_j]].value(j, q);
              int_arg += -(div_phi_u_i * phi_p_j)*stochastic_matrix[1](j_i,j_j);
            }
          }
          else{
            const unsigned int j_i = comp_i - 2*npc;
            const double phi_p_i = fe_values[pres[j_i]].value(i, q);
            if (comp_j < 2 * npc){
              const unsigned int j_j = comp_j / 2;
              const Tensor<2,2> grad_phi_u_j = fe_values[vel[j_j]].gradient(j, q);
              const double div_phi_u_j  = fe_values[vel[j_j]].divergence(j, q);
              int_arg += -(div_phi_u_j * phi_p_i)*stochastic_matrix[1](j_i,j_j);
            }
          }

          local_matrix(i, j) += int_arg * fe_values.JxW(q);
          local_matrix(i, j) += int_arg_time_dep * fe_values.JxW(q);
        }
        local_rhs(i) += (rhs->value(fe_values.quadrature_point(q))) * fe_values.JxW(q);
      }
    }
    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, system_matrix_s,system_rhs_s);
  }
}
void StationaryCoanda::solve_system() {
  //assemble_system(true);
  //system_rhs_s=0;
  //constraints.distribute(system_rhs_s);

  SolverControl solver_control(10000, 1e-6, false, false);
  //SolverCG<VectorType> direct_solver(solver_control);
  SparseDirectUMFPACK direct_solver;
  //SolverGMRES<VectorType> direct_solver(solver_control);

  NonlinearSolverSelector<VectorType>::AdditionalData data;
  data.solver_type = solver_type;
  data.strategy = strategy;
  data.maximum_non_linear_iterations = max_iter;
  data.function_tolerance = function_tolerance;
  NonlinearSolverSelector<VectorType> solver(data);

  solution= init_solution;
  if (n_blocks_load>0) {
    VectorType full_solution;
    full_solution.reinit(dofs_per_block_init);
    std::ifstream in("init");
    full_solution.block_read(in);
    in.close();
    for (unsigned int j = 0; j < n_blocks_load; ++j) {
      solution.block(j)=full_solution.block(j);
      solution.block(j+npc)=full_solution.block(j+npc_init);
    }
  }

  constraints.distribute(solution);
  //compute_static_Jacobian();
  compute_time_dep_Jacobian(solution);
  //Mp.print(std::cout);
  current_solution.reinit(solution);

  // -------------------------reinit-----------------------------------------
  solver.reinit_vector = [&](VectorType &X) {
    std::cout<<"Init"<<std::endl;
    X.reinit(dofs_per_block_s);
  };
  
  // -------------------------residual-----------------------------------------
  solver.residual = [&](const VectorType &X, VectorType &F) -> void {
    std::cout<<"Residual"<<std::endl;
    //assemble_system(true);
    //system_matrix_s.vmult(F, X);
    //F.add(-1, system_rhs_s);
    // for (unsigned int b = 0; b < solution.n_blocks(); ++b){
    //   std::cout << "----F Block " << b << " ----\n";
    //   F.block(b).print(std::cout, 4);  // 5 = precisione
    // }
    compute_residual(X,F);
    // for (unsigned int b = 0; b < solution.n_blocks(); ++b){
    //   std::cout << "----F Block " << b << " ----\n";
    //   F.block(b).print(std::cout, 4);  // 5 = precisione
    // }
  };

  // -------------------------setup_jacobian-----------------------------------------
  solver.setup_jacobian = [&](const VectorType &X) -> void {
    std::cout<<"setup_jacobian"<<std::endl;
    // for (unsigned int b = 0; b < X.n_blocks(); ++b){
    //   std::cout << "----X Block " << b << " ----\n";
    //   X.block(b).print(std::cout, 4);  // 5 = precisione
    // }
    // for (unsigned int b = 0; b < solution.n_blocks(); ++b){
    //   std::cout << "----Solution Block " << b << " ----\n";
    //   solution.block(b).print(std::cout, 4);  // 5 = precisione
    // }
    
    //assemble_system(true);
    compute_time_dep_Jacobian(X);// and also initialize the preconditioner of A
    //constraints.condense(system_matrix_s);
  };

  // -------------------------solve_with_jacobian-----------------------------------------
  solver.solve_with_jacobian = [&](const VectorType &rhs, VectorType & dst, const double tolerance) {
    std::cout<<"solve with jacobian"<<std::endl;
    //PreconditionIdentity preconditioner;
    //preconditioner.initialize(system_matrix_s);

    //BlockSchurPreconditioner<double,SparseILU<double>,PreconditionSSOR<SparseMatrix<double>>> preconditioner(system_matrix_s, A_prec, Mp_prec,system_matrix_s.block(0,1), system_matrix_s.block(1,0),Mp);
    std::cout<<"before solver"<<std::endl;
    direct_solver.initialize(system_matrix_s);
    direct_solver.vmult(dst, rhs);  // in solve_with_jacobian
    //direct_solver.solve(system_matrix_s, dst, rhs, preconditioner);
    std::cout<<"after solver"<<std::endl;
    //zero_constraints.distribute(dst);
    
    //std::vector<std::string> solution_names(2,"velocity");
    //solution_names.emplace_back("pressure");

    //std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(2, DataComponentInterpretation::component_is_part_of_vector);
    //data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  
    // DataOut<2> data_out;
    // data_out.attach_dof_handler(dof_handler);
    // data_out.add_data_vector(dst, solution_names, DataOut<2>::type_dof_data, data_component_interpretation);
    // data_out.build_patches();
    // std::ofstream output("dst.vtu");
    // data_out.write_vtu(output);
  };
  
  solver.solve(solution);
  constraints.distribute(solution);
  std::ofstream out("init");
  solution.block_write(out);
  out.close();
}
void StationaryCoanda::run(std::string input_file){
  StationaryCoanda::initialize(input_file);
  //make_grid();                                                         
  upload_grid();
  {
  dealii::TimerOutput::Scope t(timer, "setup_and_solve");
  setup_system();
  //assemble_system(false);
  solve_system();
  }
  timer_file << "N_PC = " << npc << " n_dof = "<< dof_handler.n_dofs();
  timer_file << std::endl;
  timer.print_summary();
  timer_file.flush();
  output_results();
}
void StationaryCoanda::output_results() {
  std::vector<std::string> solution_mean_names(2,"velocity_mean");
  solution_mean_names.emplace_back("pressure_mean");
  std::vector<std::string> solution_var_names(2,"velocity_var");
  solution_var_names.emplace_back("pressure_var");

  std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation_mean(2, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation_mean.push_back(DataComponentInterpretation::component_is_scalar);
  std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation_var(2, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation_var.push_back(DataComponentInterpretation::component_is_scalar);
  
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  VectorType solution_mean_out;
  solution_mean_out.reinit(dofs_per_block);
  solution_mean_out.block(0) = solution.block(0);
  solution_mean_out.block(1) = solution.block(npc);
  data_out.add_data_vector(solution_mean_out, solution_mean_names, DataOut<2>::type_dof_data, data_component_interpretation_mean);

  VectorType solution_var_out;
  solution_var_out.reinit(dofs_per_block);
  for (unsigned int j = 1; j < npc; ++j) {
    Vector<double> tmp;
    tmp = solution.block(j);
    tmp.scale(tmp);
    tmp*=norms[j];
    solution_var_out.block(0) += tmp;

    tmp = solution.block(npc+j);
    tmp.scale(tmp);
    tmp*=norms[j];
    solution_var_out.block(1) += tmp;
  }
  data_out.add_data_vector(solution_var_out, solution_var_names, DataOut<2>::type_dof_data, data_component_interpretation_var);   
  data_out.build_patches();
  int dotPosition = output_file.find('.');
  std::ofstream output(output_file.substr(0, dotPosition)+output_file.substr(dotPosition));
  data_out.write_vtu(output);  
}

int main(int argc, char** argv){
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  std::string input_file(argv[1]);
  deallog.depth_console(3);
  StationaryCoanda coanda;
  coanda.run(input_file);
  return 0;
}
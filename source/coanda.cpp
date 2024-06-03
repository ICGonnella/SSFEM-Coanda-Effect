#include <deal.II/lac/affine_constraints.h>
 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
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
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/fe/mapping_fe.h>

#include <stdexcept>
#include "boundary_conditions.hpp"
#include "utils.hpp"
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_MPI(const std::vector<std::string>& args) {
    std::vector<char*> argv_raw;
    for (const auto& arg : args) {
        argv_raw.push_back(const_cast<char*>(arg.c_str()));
    }

    int argc = static_cast<int>(args.size());
    char** argv_array = new char*[argc];
    for (int i = 0; i < argc; ++i) {
        argv_array[i] = argv_raw[i];
    }

    static Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv_array, 1);

    delete[] argv_array;
}


class StationaryCoanda
{
public:
  StationaryCoanda(int fe_degree_,  bool upload_, int n_glob_ref_, std::string mesh_input_file_, double viscosity_,double var_, int N_PC_, bool load_initial_guess_, int n_blocks_to_load_,\
		   std::string output_file_, bool verbose_);

  std::vector<SparseMatrix<double>> stochastic_matrix;
  int N_PC;
  MatrixType solution;
  VectorType variance_vec;
  
  std::vector<Point<2>> get_support_points();
  int get_dof_u() const;
  int get_dof_p() const;
  void init();
  void solve_system(int max_iter, std::string solver_type, std::string strategy, double abs_tolerance, double rel_tolerance, std::string NonlinearSolver, std::string direction_Method, std::string direction_SD_ScalingType, std::string linesearch_Method,double linesearch_FS_FullStep, double linesearch_BT_MinimumStep, double linesearch_BT_DefaultStep, double linesearch_BT_RecoveryStep, double linesearch_BT_MaximumStep, double linesearch_BT_Reduction_Factor);
  void output_results();
  std::vector<dealii::Tensor<1, 2>> evaluate_v(std::vector<Point<2>> p, VectorType func);
  std::vector<double> evaluate_p(std::vector<Point<2>> p, VectorType func);

protected:
  MPI_Comm mpi_communicator;
  void make_grid();
  void setup_system();
  void upload_grid();
  void initialize();
  void assemble();
  void collect_X(const VectorTypeMPI& X, MatrixType& X_mat);
  void distribute_F(const MatrixType& F_mat, VectorTypeMPI& F);
  void compute_linear_residual(const VectorTypeMPI& X, VectorTypeMPI& F);
  void compute_nonlinear_residual(const VectorTypeMPI& X, VectorTypeMPI& F);
  void compute_jacobian(const VectorTypeMPI& X);
  void assign_bc_system_mat(AffineConstraints<double>& constraints);

  // ----------TO INITIALIZE----------
  int fe_degree;  
  bool upload;
  int n_glob_ref;
  std::string mesh_input_file;
  double viscosity;
  double var;
  bool load_initial_guess;
  int n_blocks_to_load;
  std::string output_file;
  bool verbose;

  // ----------GRID----------
  parallel::distributed::Triangulation<2> triangulation;
  Triangulation<2> serial_triangulation;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  IndexSet locally_owned_dofs0;
  std::vector<IndexSet> block_locally_owned_dofs;
  std::vector<IndexSet> block_locally_relevant_dofs;
  IndexSet locally_owned_dofs_PC;
  IndexSet locally_relevant_dofs_PC;
  std::vector<IndexSet> block_locally_owned_dofs_PC;
  std::vector<IndexSet> block_locally_owned_dofs_PC_total;
  std::vector<IndexSet> block_locally_relevant_dofs_PC;
  unsigned int dof_u;
  unsigned int dof_p;
  std::vector<IndexSet> block_locally_owned_dofs_stochastic;
  std::vector<IndexSet> block_locally_owned_dofs_stochastic_mat;

  // ----------DOF HANDLER----------
  std::unique_ptr<FESystem<2>> fe_PC;
  std::unique_ptr<FESystem<2>> fe;
  std::unique_ptr<FESystem<2>> fe_out;
  DoFHandler<2>    dof_handler_PC;
  DoFHandler<2>    dof_handler;
  DoFHandler<2>    serial_dof_handler;
  DoFHandler<2>    serial_dof_handler_PC;
  std::vector<types::global_dof_index> dofs_per_block_PC;
  std::vector<Point<2>> support_points;

  // ----------RHS----------
  std::map<std::string, double> constants;
  std::unique_ptr<FunctionParser<2>> rhs;

  // ----------CONSTRAINTS----------
  AffineConstraints<double> constraints_PC;

  // ----------SPARSITY PATTERN----------
  BlockSparsityPattern sparsity_pattern_PC;
  BlockSparsityPattern sparsity_pattern_dense;
  SparsityPattern sparsity_pattern_stochastic;
  BlockSparsityPattern sparsity_pattern_PC_total;
  BlockDynamicSparsityPattern dsp_solution;

  // ----------MATRICES AND VECTORS----------
  MatrixTypeMPI system_matrix_L;
  std::vector<SparseMatrix<double>> system_matrix_NL;
  VectorTypeMPI solution_vec;
  VectorType serial_solution_vec;
  VectorTypeMPI system_rhs_vec;
  MatrixTypeMPI jacobian;

  // ----------------STREAMER-----------------
  ConditionalOStream pcout;

  // ----------------TIMER----------------- 
  TimerOutput computing_timer;

  //-------------EVALUATE FE FUNCTION--------------
  Utilities::MPI::RemotePointEvaluation<2> rpe;
};

StationaryCoanda::StationaryCoanda(int fe_degree_,  bool upload_, int n_glob_ref_, std::string mesh_input_file_, double viscosity_,double var_, int N_PC_, bool load_initial_guess_, int n_blocks_to_load_,std::string output_file_, bool verbose_)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator, typename Triangulation<2>::MeshSmoothing(Triangulation<2>::smoothing_on_refinement | Triangulation<2>::smoothing_on_coarsening))
  , dof_handler_PC(triangulation)
  , dof_handler(triangulation)
  , serial_dof_handler(serial_triangulation)
  , serial_dof_handler_PC(serial_triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
		    pcout,
		    TimerOutput::never,
		    TimerOutput::wall_times)
{
  fe_degree = fe_degree_;
  upload = upload_;
  n_glob_ref = n_glob_ref_;
  mesh_input_file = mesh_input_file_;
  viscosity = viscosity_;
  var = var_;
  N_PC = N_PC_;
  load_initial_guess = load_initial_guess_;
  n_blocks_to_load = n_blocks_to_load_;
  output_file = output_file_;
  verbose=verbose_;
}

std::vector<Point<2>> StationaryCoanda::get_support_points() {
  MappingQ<2> mapping(fe_degree+1);
  std::map<types::global_dof_index, Point<2>> dof_location_map = DoFTools::map_dofs_to_support_points(mapping, serial_dof_handler_PC);
  support_points.clear();
  for (auto it = dof_location_map.begin(); it != dof_location_map.end(); ++it) 
    support_points.push_back(it->second);
  return support_points;
}

int StationaryCoanda::get_dof_p() const{
  return dof_p;
}

int StationaryCoanda::get_dof_u() const{
  return dof_u;
}

std::vector<double> StationaryCoanda::evaluate_p(std::vector<Point<2>> p, VectorType func) {
  DoFHandler<2> dof_handler_p(serial_triangulation);
  FESystem<2> fe_p(FE_Q<2>(fe_degree), 1);
  dof_handler_p.distribute_dofs(fe_p);
  std::vector<unsigned int> block_component_p(1,0);
  DoFRenumbering::component_wise(dof_handler_p, block_component_p);
  MappingQ<2> mapping_p(fe_degree);
  rpe.reinit(p, serial_triangulation, mapping_p);
  return VectorTools::point_values<1>(rpe, dof_handler_p, func);
}

std::vector<dealii::Tensor<1, 2>> StationaryCoanda::evaluate_v(std::vector<Point<2>> p, VectorType func) {
  DoFHandler<2> dof_handler_u(serial_triangulation);
  FESystem<2> fe_u(FE_Q<2>(fe_degree+1), 2);
  dof_handler_u.distribute_dofs(fe_u);
  std::vector<unsigned int> block_component_u(2,0);
  DoFRenumbering::component_wise(dof_handler_u, block_component_u);
  MappingQ<2> mapping_u(fe_degree+1);
  rpe.reinit(p, serial_triangulation, mapping_u);
  return VectorTools::point_values<2>(rpe, dof_handler_u, func);
}

void StationaryCoanda::init() {
  initialize();
  if (upload)
    upload_grid();
  else
    make_grid();
  setup_system();
}

void StationaryCoanda::initialize(){
  fe_PC = std::make_unique<FESystem<2>>(FE_Q<2>(fe_degree + 1), 2*N_PC, FE_Q<2>(fe_degree), 1*N_PC);
  fe = std::make_unique<FESystem<2>>(FE_Q<2>(fe_degree + 1), 2, FE_Q<2>(fe_degree), 1);
  fe_out = std::make_unique<FESystem<2>>(FE_Q<2>(fe_degree + 1), 2*2, FE_Q<2>(fe_degree), 1*2);
  rhs = std::make_unique<FunctionParser<2>>(1);
  rhs->initialize(rhs->default_variable_names(), "0", constants);
}

void StationaryCoanda::upload_grid(){
  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation.get_triangulation());
  std::ifstream mesh_file(mesh_input_file);
  grid_in.read_msh(mesh_file);
  mesh_file.close();

  for (const auto &face : triangulation.active_face_iterators()) {
    if (face->at_boundary()) {
      if (std::fabs(face->center()[0]) < 1e-12){             // Inlet boundary
	face->set_boundary_id(1);
      }
      else if (std::fabs(face->center()[0] - 50.0) < 1e-12) // Outer boundary
	face->set_boundary_id(2);
      else                                                  // Wall boundary
	face->set_boundary_id(3);
    }
  }

  GridIn<2> grid_in2;
  grid_in2.attach_triangulation(serial_triangulation);
  std::ifstream mesh_file2(mesh_input_file);
  grid_in2.read_msh(mesh_file2);
  mesh_file2.close();

  for (const auto &face : serial_triangulation.active_face_iterators()) {
    if (face->at_boundary()) {
      if (std::fabs(face->center()[0]) < 1e-12){             // Inlet boundary
	face->set_boundary_id(1);
      }
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

  GridGenerator::create_triangulation_with_removed_cells(rectangle, cells_to_remove, serial_triangulation);
  for (const auto &face : serial_triangulation.active_face_iterators()) {
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
  // REORDER DOFS FOR DOF_HANDLER_PC
  dof_handler_PC.distribute_dofs(*fe_PC);
  dof_handler.distribute_dofs(*fe);
  serial_dof_handler.distribute_dofs(*fe);
  serial_dof_handler_PC.distribute_dofs(*fe_PC);
  std::vector<unsigned int> block_component_PC(3*N_PC,0);
  for (int i=0;i<N_PC;i++){ 
    block_component_PC[i*2] = i;
    block_component_PC[i*2+1] = i;
    block_component_PC[2*N_PC+i] = N_PC+i;
  }
  std::vector<unsigned int> block_component(3,0);
  block_component[0] = 0;
  block_component[1] = 0;
  block_component[2] = 1;
  DoFRenumbering::component_wise(dof_handler_PC, block_component_PC);
  DoFRenumbering::component_wise(dof_handler, block_component);
  DoFRenumbering::component_wise(serial_dof_handler, block_component);
  DoFRenumbering::component_wise(serial_dof_handler_PC, block_component_PC);
  
  dofs_per_block_PC = DoFTools::count_dofs_per_fe_block(dof_handler_PC, block_component_PC);
  dof_u = dofs_per_block_PC[0];
  dof_p = dofs_per_block_PC[N_PC];
  pcout << "Number of active cells: "<< triangulation.n_active_cells()<< std::endl << "Number of degrees of freedom: " << dof_handler_PC.n_dofs() << " = ("<<dof_u << " + "<< dof_p <<")*"<<N_PC<<std::endl;
  
  // SETUP CONSTRAINTS
  {
  constraints_PC.clear();
  DoFTools::make_hanging_node_constraints(serial_dof_handler_PC, constraints_PC);
  const FEValuesExtractors::Vector velocities_PC(0*2);
  VectorTools::interpolate_boundary_values(serial_dof_handler_PC, 1, DirichletBoundary1(N_PC), constraints_PC, fe_PC->component_mask(velocities_PC));
  VectorTools::interpolate_boundary_values(serial_dof_handler_PC, 3, DirichletBoundary3(N_PC), constraints_PC, fe_PC->component_mask(velocities_PC));
  }
  constraints_PC.close();

  // LOCAL DOFS
  locally_owned_dofs_PC = dof_handler_PC.locally_owned_dofs();
  locally_relevant_dofs_PC = DoFTools::extract_locally_relevant_dofs(dof_handler_PC);
  for (int i=0; i<N_PC; i++) {
    block_locally_owned_dofs_PC.push_back(IndexSet(dof_u));
    block_locally_relevant_dofs_PC.push_back(IndexSet(dof_u));
    block_locally_owned_dofs_PC_total.push_back(IndexSet(dof_u));
  }
  for (int i=0; i<N_PC; i++) {
    block_locally_owned_dofs_PC.push_back(IndexSet(dof_p));
    block_locally_relevant_dofs_PC.push_back(IndexSet(dof_p));
    block_locally_owned_dofs_PC_total.push_back(IndexSet(dof_p));
  }
  for (unsigned int i=0; i<(dof_u+dof_p)*N_PC; i++) {
    int iq = i/(dof_u*N_PC);
    int ir = i%(dof_u*N_PC);
    block_locally_owned_dofs_PC_total[iq*N_PC + (ir/dof_p)*iq + (ir/dof_u) - (ir/dof_u)*iq].add_index(ir- iq*(ir/dof_p)*dof_p - (ir/dof_u)*dof_u + iq*(ir/dof_u)*dof_u);
  }
  for (const auto &i : locally_owned_dofs_PC) {
    int iq = i/(dof_u*N_PC);
    int ir = i%(dof_u*N_PC);
    block_locally_owned_dofs_PC[iq*N_PC + (ir/dof_p)*iq + (ir/dof_u) - (ir/dof_u)*iq].add_index(ir- iq*(ir/dof_p)*dof_p - (ir/dof_u)*dof_u + iq*(ir/dof_u)*dof_u);
  }
  for (const auto &i : locally_owned_dofs_PC) {
    int iq = i/(dof_u*N_PC);
    int ir = i%(dof_u*N_PC);
    block_locally_relevant_dofs_PC[iq*N_PC + (ir/dof_p)*iq + (ir/dof_u) - (ir/dof_u)*iq].add_index(ir - iq*(ir/dof_p)*dof_p - (ir/dof_u)*dof_u + iq*(ir/dof_u)*dof_u);
  }
  
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  locally_owned_dofs0 = IndexSet(dof_u);
  for (const auto &i : locally_owned_dofs) {
    if (i<dof_u)
      locally_owned_dofs0.add_index(i);
  }
  block_locally_owned_dofs.push_back(IndexSet(dof_u));
  block_locally_owned_dofs.push_back(IndexSet(dof_p));
  block_locally_relevant_dofs.push_back(IndexSet(dof_u));
  block_locally_relevant_dofs.push_back(IndexSet(dof_p));
  for (const auto &i : locally_owned_dofs) {
    int iq = i/(dof_u);
    block_locally_owned_dofs[iq].add_index(i - iq*dof_u);
  }
  for (const auto &i : locally_relevant_dofs) {
    int iq = i/(dof_u);
    block_locally_relevant_dofs[iq].add_index(i - iq*dof_u);
  }
  
  block_locally_owned_dofs_stochastic.push_back(IndexSet(N_PC));
  for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(mpi_communicator); i++)
    block_locally_owned_dofs_stochastic_mat.push_back(IndexSet(N_PC));
  for (int i=0; i<N_PC; i++) {
    block_locally_owned_dofs_stochastic[0].add_index(i);
    block_locally_owned_dofs_stochastic_mat[Utilities::MPI::this_mpi_process(mpi_communicator)].add_index(i);
  }
  
  
  // SETUP USEFUL SPARSITY PATTERNS (sparsity_pattern_stochastic, sparsity_pattern_dense, sparsity_pattern_PC)
  
  DynamicSparsityPattern dsp_stochastic(N_PC, N_PC);
  for (int i=0;i<N_PC*N_PC;i++)
    dsp_stochastic.add(i/N_PC,i%N_PC);

  std::vector<unsigned int> n_elem_NPC_mat;
  for (unsigned int i=0;i<Utilities::MPI::n_mpi_processes(mpi_communicator);i++)
    n_elem_NPC_mat.push_back(N_PC);
  std::vector<unsigned int> n_elem_NPC;
  n_elem_NPC.push_back(N_PC);
  BlockDynamicSparsityPattern dsp_stochastic_mat(n_elem_NPC_mat, n_elem_NPC);
  for (unsigned int i=0;i<(N_PC*Utilities::MPI::n_mpi_processes(mpi_communicator))*N_PC;i++)
    dsp_stochastic_mat.add(i/N_PC,i%N_PC);

  BlockDynamicSparsityPattern dsp_PC(dofs_per_block_PC, dofs_per_block_PC);
  DoFTools::make_sparsity_pattern(serial_dof_handler_PC, dsp_PC, constraints_PC,true);
  SparsityTools::distribute_sparsity_pattern(dsp_PC, dof_handler_PC.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs_PC);
  sparsity_pattern_PC.copy_from(dsp_PC);

  BlockDynamicSparsityPattern dsp_PC_total(dofs_per_block_PC, dofs_per_block_PC);
  DoFTools::make_sparsity_pattern(serial_dof_handler_PC, dsp_PC_total, constraints_PC,true);
  sparsity_pattern_PC_total.copy_from(dsp_PC_total);

  std::vector<unsigned int> n_elem_solution;
  n_elem_solution.push_back(dof_u);
  n_elem_solution.push_back(dof_p);
  dsp_solution.reinit(n_elem_solution, n_elem_NPC);
  for (unsigned int i=0;i<(dof_u+dof_p)*N_PC;i++)
    dsp_solution.add(i/N_PC,i%N_PC);

  sparsity_pattern_dense.copy_from(dsp_solution);
  sparsity_pattern_stochastic.copy_from(dsp_stochastic);

  // SETUP MATRICES AND VECTORS

  //-----------------------system_matrix_L---------------------------
  system_matrix_L.reinit(block_locally_owned_dofs_PC, block_locally_owned_dofs_PC_total, dsp_PC, mpi_communicator);
  system_matrix_L.compress(VectorOperation::insert);
  
  //-----------------------system_matrix_NL---------------------------
  for (unsigned int n=0;n<dof_u;n++){
    system_matrix_NL.push_back(SparseMatrix<double>());
    if (locally_owned_dofs.is_element(n))
      system_matrix_NL[n].reinit(sparsity_pattern_PC.block(0,0));
  }

  //-----------------------jacobian---------------------------
  jacobian.reinit(block_locally_owned_dofs_PC, block_locally_owned_dofs_PC_total, dsp_PC, mpi_communicator);
  jacobian.compress(VectorOperation::insert);

  //-----------------------stochastic_matrix---------------------------
  stochastic_matrix.resize(N_PC+2);
  for (int i=0;i<N_PC+2;i++) {
    stochastic_matrix[i].reinit(sparsity_pattern_stochastic);
  }

  //-----------------------solution---------------------------
  std::vector<types::global_dof_index> dofs_per_block;
  dofs_per_block.push_back(dofs_per_block_PC[0]);
  dofs_per_block.push_back(dofs_per_block_PC[N_PC]);
  solution.reinit(sparsity_pattern_dense);
  solution_vec.reinit(block_locally_owned_dofs_PC, mpi_communicator);
  serial_solution_vec.reinit(dofs_per_block_PC);
  //-----------------------rhs---------------------------
  system_rhs_vec.reinit(block_locally_owned_dofs_PC, mpi_communicator);

  //-----------------------variance---------------------------
  variance_vec.reinit(dofs_per_block);
}

void StationaryCoanda::assemble() {
  TimerOutput::Scope t(computing_timer, "assemble");
  system_matrix_L *= 0;
  system_matrix_L.compress(VectorOperation::add);
  for (unsigned int n=0; n<dof_u; n++) {
    if (locally_owned_dofs.is_element(n))
      system_matrix_NL[n]*=0;
  }
  system_rhs_vec = 0;
  QGauss<2> quadrature_formula(fe_degree + 2);
  FEValues<2> fe_values(*fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();
  
  std::vector<FullMatrix<double>> local_matrix;
  local_matrix.emplace_back(FullMatrix<double>(dofs_per_cell, dofs_per_cell));
  for (unsigned int i=0; i<dofs_per_cell; i++)
    local_matrix.emplace_back(FullMatrix<double>(dofs_per_cell, dofs_per_cell));
  Vector<double> local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, 2>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, 2>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(2);
  for (const auto &cell : serial_dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    local_matrix[0] = 0;
    for (unsigned int n=0; n<dofs_per_cell; n++)
      local_matrix[1+n] = 0;
    local_rhs    = 0;
    
    for (unsigned int q = 0; q < n_q_points; ++q) {
      for (unsigned int k = 0; k < dofs_per_cell; ++k) {
        div_phi_u[k]  = fe_values[velocities].divergence(k, q);
        grad_phi_u[k] = fe_values[velocities].gradient(k, q);
        phi_u[k]      = fe_values[velocities].value(k, q);
	phi_p[k]      = fe_values[pressure].value(k, q);
      }
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
	local_rhs(i) += (rhs->value(fe_values.quadrature_point(q))) * fe_values.JxW(q);
	for (unsigned int j = 0; j < dofs_per_cell; ++j) {
	  local_matrix[0](i, j) += (scalar_product(grad_phi_u[i], grad_phi_u[j])-
				    phi_p[i] * div_phi_u[j]-
				    div_phi_u[i] * phi_p[j] +
				    phi_p[i] * phi_p[j]) * fe_values.JxW(q);
	  for (unsigned int l = 0; l<dofs_per_cell; ++l){
	    local_matrix[1+l](i, j) += ((grad_phi_u[j]*phi_u[i])*phi_u[l]) * fe_values.JxW(q);
	  }
	}
      }
    }
    cell->get_dof_indices(local_dof_indices);
    for (const unsigned int l : fe_values.dof_indices()) {
      for (const unsigned int i : fe_values.dof_indices()) {
	if (l==0 && locally_owned_dofs.is_element(local_dof_indices[i])) {
	  std::vector<unsigned int> idx(1, local_dof_indices[i]%dof_u + (local_dof_indices[i]/dof_u)*dof_u*N_PC);
	  std::vector<unsigned int> v(1, local_rhs(i));
	  system_rhs_vec.add(idx, v);
	  for (const unsigned int j : fe_values.dof_indices()) {
	    system_matrix_L.add(local_dof_indices[i]%dof_u + (local_dof_indices[i]/dof_u)*dof_u*N_PC, local_dof_indices[j]%dof_u + (local_dof_indices[j]/dof_u)*dof_u*N_PC, local_matrix[0](i, j));
	  }
	}
	if (locally_owned_dofs.is_element(local_dof_indices[l])) {
	  if (local_dof_indices[l]<dof_u) {
	    for (const unsigned int j : fe_values.dof_indices())
	      system_matrix_NL[local_dof_indices[l]].add(local_dof_indices[i],local_dof_indices[j],local_matrix[1+l](i, j));
	  }
	  else
	    if(local_matrix[1+l].all_zero()==false)
	      std::cout<<local_dof_indices[l]<<" INFORMATION LOST"<<std::endl;
	}
      }
    }
  }
  system_matrix_L.compress(VectorOperation::add);
  system_rhs_vec.compress(VectorOperation::add);
  assign_bc_system_mat(constraints_PC);
  system_matrix_L.compress(VectorOperation::insert);
  system_rhs_vec.compress(VectorOperation::insert);
  return;
}


void StationaryCoanda:: compute_linear_residual(const VectorTypeMPI& X, VectorTypeMPI& F) {
  MatrixType X_mat(sparsity_pattern_dense);
  MatrixType F_mat(sparsity_pattern_dense);
  collect_X(X, X_mat);
  TimerOutput::Scope t(computing_timer, "compute_linear_residual");
  
  SparseMatrix<double> dx1(sparsity_pattern_dense.block(0,0));
  SparseMatrix<double> dx2(sparsity_pattern_dense.block(1,0));
  SparseMatrix<double> dx3(sparsity_pattern_dense.block(0,0));
  X_mat.block(0,0).mmult(dx1, stochastic_matrix[0], Vector<double>(), false);
  X_mat.block(1,0).mmult(dx2, stochastic_matrix[1], Vector<double>(), false);
  X_mat.block(0,0).mmult(dx3, stochastic_matrix[1], Vector<double>(), false);

  MatrixType bm_L;
  SparseMatrix<double> tmp_m(sparsity_pattern_dense.block(0,0));
  bm_L.reinit(sparsity_pattern_PC);
  matMPI_to_mat(system_matrix_L, bm_L.block(0,0), locally_owned_dofs_PC, 0, dof_u, 0);
  matMPI_to_mat(system_matrix_L, bm_L.block(0,N_PC), locally_owned_dofs_PC, 0, dof_u, dof_u*N_PC);
  matMPI_to_mat(system_matrix_L, bm_L.block(N_PC,0), locally_owned_dofs_PC, dof_u*N_PC, dof_u*N_PC+dof_p, 0);
  
  bm_L.block(0,0).mmult(F_mat.block(0,0), dx1, Vector<double>(), false);
  bm_L.block(0,N_PC).mmult(tmp_m, dx2, Vector<double>(), false);
  bm_L.block(N_PC,0).mmult(F_mat.block(1,0), dx3, Vector<double>(), false);
  
  F_mat.block(0,0).add(1, tmp_m);
  distribute_F(F_mat, F);
}

void StationaryCoanda::compute_nonlinear_residual(const VectorTypeMPI& X, VectorTypeMPI& F) {
  MatrixType X_mat(sparsity_pattern_dense);
  MatrixType F_mat(sparsity_pattern_dense);
  VectorTypeMPI F_tmp(block_locally_owned_dofs_PC, mpi_communicator);
  collect_X(X, X_mat);
  TimerOutput::Scope t(computing_timer, "compute_nonlinear_residual");
  
  DynamicSparsityPattern dsp_tmp1(1, 1);
  dsp_tmp1.add(0,0);
  DynamicSparsityPattern dsp_tmp2(1, 1);
  dsp_tmp2.add(0,0);
  SparsityPattern sp_tmp1;
  sp_tmp1.copy_from(dsp_tmp1);
  SparsityPattern sp_tmp2;
  sp_tmp2.copy_from(dsp_tmp2);
  
  
  for (unsigned int i=0;i<dof_u;i++)
    if (locally_owned_dofs0.is_element(i)) {
      SparseMatrix<double> tmp_mat_1(sparsity_pattern_PC.block(0,0));
      SparseMatrix<double> tmp_mat_res_1(sp_tmp1);
      SparseMatrix<double> tmp_mat_res_2(sp_tmp2);
      X_mat.block(0,0).Tmmult(tmp_mat_res_1, system_matrix_NL[i]);
      tmp_mat_res_1.mmult(tmp_mat_res_2, X_mat.block(0,0));
      for (int j=0;j<N_PC; j++){
	F_mat.set(i,j,Hadamard(tmp_mat_res_2, stochastic_matrix[2+j]));
      }
    }
  distribute_F(F_mat, F_tmp);
  F.add(1, F_tmp);
  F.compress(VectorOperation::add);
}

void StationaryCoanda::assign_bc_system_mat(AffineConstraints<double>& constraints) {
  TimerOutput::Scope t(computing_timer, "bc");
  for(unsigned int i=0; i<dof_u; i++){
    if(locally_owned_dofs.is_element(i)) {
      const auto constr = constraints.get_constraint_entries(i);
      if (constr!=nullptr){
	system_matrix_NL[i] = 0;
	const auto inh = constraints.get_inhomogeneity(i);
	SparsityPattern::iterator row_begin = sparsity_pattern_PC.block(0,0).begin(i);
	SparsityPattern::iterator row_end = sparsity_pattern_PC.block(0,0).end(i);
	for (dealii::SparsityPattern::iterator it = row_begin; it != row_end; ++it) {
	  PETScWrappers::MPI::BlockSparseMatrix::value_type elem = i==it->column() ? 1:0;
	  system_matrix_L.set(i, it->column(), elem);
	}
	row_begin = sparsity_pattern_PC.block(0,N_PC).begin(i);
	row_end = sparsity_pattern_PC.block(0,N_PC).end(i);
	for (dealii::SparsityPattern::iterator it = row_begin; it != row_end; ++it) {
	  int elem = 0;
	  system_matrix_L.block(0,N_PC).set(i, it->column(), elem);
	}
	for (int j=0;j<N_PC; j++) {
	  double factor = stochastic_matrix[0](0,j);
	  system_rhs_vec[(i + j*dof_u)*(1 - i/(dof_u)) + (dof_u*N_PC + j*dof_p + i%dof_u)*(i/dof_u)] = factor*inh;
	}
      }
    }
  }
}

void StationaryCoanda::collect_X(const VectorTypeMPI& X, MatrixType& X_mat) {
  TimerOutput::Scope t(computing_timer, "collect_X");
  vec_to_mat(X, X_mat, locally_owned_dofs);
  Utilities::MPI::sum(X_mat.block(0,0), mpi_communicator, X_mat.block(0,0));
  Utilities::MPI::sum(X_mat.block(1,0), mpi_communicator, X_mat.block(1,0));
}

void StationaryCoanda::distribute_F(const MatrixType& F_mat, VectorTypeMPI& F) {
  TimerOutput::Scope t(computing_timer, "distribute_F");
  mat_to_vec(F_mat, F, locally_owned_dofs);
}

void StationaryCoanda::compute_jacobian(const VectorTypeMPI& X) {
  MatrixType X_mat(sparsity_pattern_dense);
  MatrixType F_mat(sparsity_pattern_dense);
  collect_X(X, X_mat);
  TimerOutput::Scope t(computing_timer, "jacobian");

  DynamicSparsityPattern dsp_tmp(1, 1);
  dsp_tmp.add(0,0);
  SparsityPattern sp_tmp3;
  sp_tmp3.copy_from(dsp_tmp);
  std::vector<SparseMatrix<double>> tmp_3(dof_u);
  for (unsigned int i=0; i<dof_u; i++) {
    if (locally_owned_dofs0.is_element(i)) {
      tmp_3[i].reinit(sp_tmp3);
      SparseMatrix<double> sparsemat_tmp(sparsity_pattern_PC.block(0,0));
      element_copy(system_matrix_NL[i], sparsemat_tmp);
      sparsemat_tmp.symmetrize();
      sparsemat_tmp*=2;
      sparsemat_tmp.mmult(tmp_3[i], X_mat.block(0,0));
    }
  }

  for (int i=0; i<N_PC*N_PC; i++) {
    SparseMatrix<double> mat00(sparsity_pattern_PC.block(0,0));
    SparseMatrix<double> mat01(sparsity_pattern_PC.block(0,N_PC));
    SparseMatrix<double> mat10(sparsity_pattern_PC.block(N_PC,0));

    matMPI_to_mat(system_matrix_L, mat00, locally_owned_dofs_PC, 0, dof_u, 0);
    matMPI_to_mat(system_matrix_L, mat01, locally_owned_dofs_PC, 0, dof_u, dof_u*N_PC);
    matMPI_to_mat(system_matrix_L, mat10, locally_owned_dofs_PC, dof_u*N_PC, dof_u*N_PC+dof_p, 0);
    mat00*=stochastic_matrix[0](i%N_PC,i/N_PC);
    mat01*=stochastic_matrix[1](i%N_PC,i/N_PC);
    mat10*=stochastic_matrix[1](i%N_PC,i/N_PC);
    
    SparseMatrix<double> mat1(sparsity_pattern_PC.block(0,0));
    Vector<double> tmp_vec_1(N_PC);
    Vector<double> tmp_vec_5(dof_u);

    for (int idx=0;idx<N_PC;idx++){
      tmp_vec_1[idx] = stochastic_matrix[2+i/N_PC](i%N_PC, idx);
    }
    
    for (unsigned int ii=0; ii<dof_u; ii++)
      if (locally_owned_dofs0.is_element(ii)) {
	tmp_vec_5=0;
	tmp_3[ii].vmult(tmp_vec_5, tmp_vec_1);
     
	SparsityPattern::iterator row_begin = mat1.get_sparsity_pattern().begin(ii);
	SparsityPattern::iterator row_end = mat1.get_sparsity_pattern().end(ii);
	for (dealii::SparsityPattern::iterator it = row_begin; it != row_end; ++it)
	  mat1.set(ii, it->column(),tmp_vec_5[it->column()]);
      }
    mat00.add(1, mat1);
    mat_to_matMPI(mat00, jacobian.block(i/N_PC,i%N_PC), locally_owned_dofs, 0, dof_u, 0);
    mat_to_matMPI(mat01, jacobian.block(i/N_PC,N_PC+i%N_PC), locally_owned_dofs, 0, dof_u, 0);
    mat_to_matMPI(mat10, jacobian.block(i/N_PC+N_PC,i%N_PC), locally_owned_dofs, dof_u, dof_u+dof_p, 0);
    
    jacobian.block(i/N_PC+N_PC, i%N_PC+N_PC) = 0;
  }
  jacobian.compress(VectorOperation::insert);
}

void StationaryCoanda::solve_system(int max_iter, std::string solver_type, std::string strategy, double abs_tolerance, double rel_tolerance, std::string NonlinearSolver, std::string direction_Method, std::string direction_SD_ScalingType, std::string linesearch_Method,double linesearch_FS_FullStep, double linesearch_BT_MinimumStep, double linesearch_BT_DefaultStep, double linesearch_BT_RecoveryStep, double linesearch_BT_MaximumStep, double linesearch_BT_Reduction_Factor) {
  //------------------------------------------DATA--------------------------------------------------
  NonlinearSolverSelector<VectorTypeMPI>::AdditionalData data;
  if (solver_type=="automatic") 
    data.solver_type = NonlinearSolverSelector<VectorTypeMPI>::AdditionalData::SolverType::automatic;
  if (solver_type=="nox") 
    data.solver_type = NonlinearSolverSelector<VectorTypeMPI>::AdditionalData::SolverType::nox;
  if (solver_type=="petsc_snes")
    data.solver_type = NonlinearSolverSelector<VectorTypeMPI>::AdditionalData::SolverType::petsc_snes;
  if (solver_type=="kinsol")
    data.solver_type = NonlinearSolverSelector<VectorTypeMPI>::AdditionalData::SolverType::kinsol;
  if (strategy=="newton")
    data.strategy = NonlinearSolverSelector<VectorTypeMPI>::AdditionalData::SolutionStrategy::newton;
  if (strategy=="linesearch")
      data.strategy = NonlinearSolverSelector<VectorTypeMPI>::AdditionalData::SolutionStrategy::linesearch;
  if (strategy=="picard")
      data.strategy = NonlinearSolverSelector<VectorTypeMPI>::AdditionalData::SolutionStrategy::picard;
  data.maximum_non_linear_iterations = max_iter;
  data.function_tolerance = abs_tolerance;
  //------------------------------------------PARAMETERS_NOX--------------------------------------------------
  Teuchos::RCP< Teuchos::ParameterList > parameters_nox(new Teuchos::ParameterList);
  parameters_nox->set ("Nonlinear Solver", NonlinearSolver); // Line Search Based, Trust Region Based, Inexact Trust Region Based, Tensor Based 

  Teuchos::RCP< Teuchos::ParameterList > direction(new Teuchos::ParameterList);
  direction->set("Method", direction_Method); // Newton, Steepest Descent, NonlinearCG, Broyden, Tensor, Modified-Newton, Quasi-Newton, User Defined
  Teuchos::RCP< Teuchos::ParameterList > steepest_descent(new Teuchos::ParameterList); 
  steepest_descent->set("Scaling Type", direction_SD_ScalingType); // 2-Norm, F 2-Norm, None
  direction->set("Steepest Descent", *steepest_descent.get());
  parameters_nox->set ("Direction", *direction.get());
  Teuchos::RCP< Teuchos::ParameterList > linesearch(new Teuchos::ParameterList);
  linesearch->set("Method", linesearch_Method); // Full Step, Backtrack, More'-Thuente
  Teuchos::RCP< Teuchos::ParameterList > fullstep(new Teuchos::ParameterList);
  fullstep->set("Full Step", linesearch_FS_FullStep); 
  Teuchos::RCP< Teuchos::ParameterList > backtrack(new Teuchos::ParameterList);
  backtrack->set("Minimum Step", linesearch_BT_MinimumStep);
  backtrack->set("Default Step", linesearch_BT_DefaultStep);
  backtrack->set("Recovery Step", linesearch_BT_RecoveryStep);
  backtrack->set("Maximum Step", linesearch_BT_MaximumStep);
  backtrack->set("Max Iters", 100);
  backtrack->set("Reduction Factor", linesearch_BT_Reduction_Factor);
  linesearch->set("Full Step", *fullstep.get());
  linesearch->set("Backtrack", *backtrack.get());
  parameters_nox->set("Line Search", *linesearch.get());

  //------------------------------------------DATA_NOX--------------------------------------------------
  TrilinosWrappers::NOXSolver<VectorTypeMPI>::AdditionalData data_nox;
  data_nox.max_iter = max_iter;
  data_nox.abs_tol = abs_tolerance;
  data_nox.rel_tol = rel_tolerance;
  //------------------------------------------SOLVER--------------------------------------------------
  NonlinearSolverSelector<VectorTypeMPI> solver(data);
  solver.set_data(data_nox, parameters_nox);
  
  // ------residual------
  solver.residual = [&](const VectorTypeMPI &X, VectorTypeMPI &F) -> void {
    if (verbose) {
      pcout<<"Residual"<<std::endl;
      vec_to_mat(X, solution, locally_owned_dofs);
      Utilities::MPI::sum(solution.block(0,0), mpi_communicator, solution.block(0,0));
      Utilities::MPI::sum(solution.block(1,0), mpi_communicator, solution.block(1,0));
      mat_to_vec(solution, serial_solution_vec);
      output_results();
    }
    compute_linear_residual(X, F);
    compute_nonlinear_residual(X, F);
    F.add(-1, system_rhs_vec);
    F.compress(VectorOperation::add);
  };

  // ------setup_jacobian-------
  solver.setup_jacobian = [&](const VectorTypeMPI &X) -> void {
    if (verbose)
      pcout<<"setup_jacobian"<<std::endl;
    compute_jacobian(X);
  };
  
  // ------solve_with_jacobian------
  solver.solve_with_jacobian = [&](const VectorTypeMPI &rhs, VectorTypeMPI & dst, const double tolerance) {
    if (verbose)
      pcout<<"solve with jacobian with tolerance "<<tolerance<<std::endl;
    BlockSparseMatrix<double> j(sparsity_pattern_PC_total);
    MatrixType rhs_mat;
    rhs_mat.reinit(sparsity_pattern_dense);
    BlockVector<double> dst_;
    dst_.reinit(dofs_per_block_PC);
    vec_to_mat(rhs,rhs_mat, locally_owned_dofs);
    Utilities::MPI::sum(rhs_mat.block(0,0), mpi_communicator, rhs_mat.block(0,0));
    Utilities::MPI::sum(rhs_mat.block(1,0), mpi_communicator, rhs_mat.block(1,0));
    for (int i=0; i<2*N_PC*2*N_PC; i++) {
      matMPI_to_mat(jacobian.block(i/(N_PC*2),i%(N_PC*2)), j.block(i/(N_PC*2),i%(N_PC*2)), locally_owned_dofs, dof_u*(i/(N_PC*2*N_PC)), dof_u*(1-i/(N_PC*2*N_PC))+(dof_u+dof_p)*(i/(N_PC*2*N_PC)));
      Utilities::MPI::sum(j.block(i/(N_PC*2),i%(N_PC*2)), mpi_communicator, j.block(i/(N_PC*2),i%(N_PC*2)));
    }
    SparseDirectUMFPACK direct_solver;
    mat_to_vec(rhs_mat, dst_);
    try {
      direct_solver.solve(j, dst_);
    } catch(std::exception& e) {}
    vec_to_vecMPI(dst_, dst, locally_owned_dofs_PC);
  };
  
  assemble();
  if (load_initial_guess) {
    read_blockvector(serial_solution_vec, "initial_guess", n_blocks_to_load);
    vec_to_vecMPI(serial_solution_vec, solution_vec, locally_owned_dofs_PC);
    solution_vec.compress(VectorOperation::insert);
  }
  else {
    solution_vec=0;
    solution_vec.compress(VectorOperation::insert);
  }
  solver.solve(solution_vec);
  vec_to_mat(solution_vec, solution, locally_owned_dofs);
  Utilities::MPI::sum(solution.block(0,0), mpi_communicator, solution.block(0,0));
  Utilities::MPI::sum(solution.block(1,0), mpi_communicator, solution.block(1,0));
  mat_to_vec(solution, serial_solution_vec);
  if(verbose) {
    computing_timer.print_summary();
    computing_timer.reset();
  }
}

void StationaryCoanda::output_results() {

  DoFHandler<2> dof_handler_out(serial_triangulation);
  dof_handler_out.distribute_dofs(*fe_out);
  std::vector<unsigned int> block_component_out(3*2,0);
  for (unsigned int i=0;i<2;i++){
    block_component_out[i*2] = i;
    block_component_out[i*2+1] = i;
    block_component_out[2*2+i] = 2 + i;
  }
  DoFRenumbering::component_wise(dof_handler_out, block_component_out);
  
  if (Utilities::MPI::this_mpi_process(mpi_communicator)==0) {
    write_blockvector(serial_solution_vec,"initial_guess", serial_solution_vec.n_blocks());
    
    std::vector<std::string> solution_names(2,"velocity");
    solution_names.emplace_back("velocity_variance");
    solution_names.emplace_back("velocity_variance");
    solution_names.emplace_back("pressure");
    solution_names.emplace_back("pressure_variance");
  
    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(2, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    
    std::vector<types::global_dof_index> dofs_per_block_dataout;
    dofs_per_block_dataout.push_back(dof_u);
    dofs_per_block_dataout.push_back(dof_u);
    dofs_per_block_dataout.push_back(dof_p);
    dofs_per_block_dataout.push_back(dof_p);
    VectorType solution_out(dofs_per_block_dataout);
    solution_out.block(0) = serial_solution_vec.block(0);
    solution_out.block(1) = variance_vec.block(0);
    solution_out.block(2) = serial_solution_vec.block(N_PC);
    solution_out.block(3) = variance_vec.block(1);
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler_out);
    data_out.add_data_vector(solution_out, solution_names, DataOut<2>::type_dof_data, data_component_interpretation);    
    data_out.build_patches();
    int dotPosition = output_file.find('.');
    std::ofstream output(output_file.substr(0, dotPosition)+output_file.substr(dotPosition));
    data_out.write_vtu(output);
  }
}

// -------------------------------------------------------------------BINDINGS-------------------------------------------------------------------------

PYBIND11_MODULE(libcoanda, m) {
    m.doc() = "ssfem implementation based n dealii";

    m.def("init_MPI", &init_MPI, "A function that init MPI");
    
    py::class_<StationaryCoanda>(m, "StationaryCoanda")
      .def(py::init<int&,
	   bool&,
	   int&,
	   std::string&,
	   double&,
	   double&,
	   int&,
	   bool&,
	   int&,
	   std::string&,
	   bool&>(),
	   py::arg("fe_degree")=1,
	   py::arg("upload")=true,
	   py::arg("n_glob_ref")=3,
	   py::arg("mesh_input_file")=std::string("../parameters/mesh18o1.msh"),
	   py::arg("viscosity")=0.9,
	   py::arg("var")=0.01,
	   py::arg("N_PC")=1,

	   py::arg("load_initial_guess")=false,
	   py::arg("n_blocks_to_load")=1,
	   py::arg("output_file")=std::string("coanda.vtu"),
	   py::arg("verbose")=true
	   ) 

      .def("initialize",
	   [](StationaryCoanda &obj) {
	     obj.init();
	   })

      .def("get_point_val",
	   [](StationaryCoanda &obj, py::array_t<double, py::array::c_style | py::array::forcecast> points, py::array_t<double, py::array::c_style | py::array::forcecast> func, int idx) {

	     auto buffer_info_points = points.request();
	     size_t rows_points = buffer_info_points.shape[0];
	     size_t cols_points = buffer_info_points.shape[1];
	     auto *ptr_points = static_cast<double *>(buffer_info_points.ptr);
	     std::vector<Point<2>> p_tmp;
	     for (unsigned int i = 0; i < rows_points; ++i)
	       p_tmp.push_back(Point<2>(ptr_points[i*2], ptr_points[i*2+1]));

	     auto buffer_info_func = func.request();
	     size_t rows_func = buffer_info_func.shape[0];
	     size_t cols_func = buffer_info_func.shape[1];
	     auto *ptr_func = static_cast<double *>(buffer_info_func.ptr);
	     VectorType func_tmp_u;
	     VectorType func_tmp_p;
	     std::vector<types::global_dof_index> dofs_per_block_u(1,obj.get_dof_u());
	     std::vector<types::global_dof_index> dofs_per_block_p(1,obj.get_dof_p());
	     func_tmp_u.reinit(dofs_per_block_u);
	     func_tmp_p.reinit(dofs_per_block_p);
	     for (unsigned int i = 0; i < rows_func; ++i) {
	       if (i<obj.get_dof_u())
		 func_tmp_u[i] = ptr_func[i*obj.N_PC+idx];
	       else
		 func_tmp_p[i-obj.get_dof_u()] = ptr_func[obj.N_PC*obj.get_dof_u() + (i-obj.get_dof_u())*obj.N_PC + idx];
	     }
	     size_t cols_v = 3;
	     py::array_t<double, py::array::c_style | py::array::forcecast> numpy_solution({rows_points, cols_v});	     
	     auto res_v = obj.evaluate_v(p_tmp, func_tmp_u);
	     auto res_p = obj.evaluate_p(p_tmp, func_tmp_p);
	     
	     auto buffer_info = numpy_solution.request();
	     double *ptr = static_cast<double *>(buffer_info.ptr);

	     for (unsigned int i = 0; i < cols_points; ++i) {
	       ptr[3*i] = res_v[i][0];
	       ptr[3*i + 1] = res_v[i][1];
	       ptr[3*i + 2] = res_p[i];
	     }
	     return numpy_solution;
     })

      .def("get_sol",
     [](StationaryCoanda &obj) {
       const unsigned int rows = obj.solution.m();
       const unsigned int cols = obj.solution.n();
       
       py::array_t<double, py::array::c_style | py::array::forcecast> numpy_solution({rows, cols});
       auto buffer_info = numpy_solution.request();
       double *ptr = static_cast<double *>(buffer_info.ptr);

       for (unsigned int i = 0; i < rows; ++i) {
	 for (unsigned int j = 0; j < cols; ++j) {
	   ptr[i * cols + j] = obj.solution(i, j);
	 }
       }
       return numpy_solution;
     })

      .def("set_stochastic_mat",
	   [](StationaryCoanda &obj, unsigned int idx, py::array_t<double, py::array::c_style | py::array::forcecast> numpy_matrix) {
	     auto buffer_info = numpy_matrix.request();
	     if (buffer_info.ndim != 2) {
	       throw std::runtime_error("Number of dimensions must be 2");
	     }
	     size_t rows = buffer_info.shape[0];
	     size_t cols = buffer_info.shape[1];
	     auto *ptr = static_cast<double *>(buffer_info.ptr);
	     for (size_t i = 0; i < rows; ++i) {
	       for (size_t j = 0; j < cols; ++j) {
                 obj.stochastic_matrix[idx].set(i, j, ptr[i * cols + j]);
	       }
	     }
	   })

      .def("set_sol_variance",
	   [](StationaryCoanda &obj, py::array_t<double, py::array::c_style | py::array::forcecast> numpy_vect) {
	     auto buffer_info = numpy_vect.request();
	     size_t rows = buffer_info.shape[0];
	     auto *ptr = static_cast<double *>(buffer_info.ptr);
	     for (size_t i = 0; i < rows; ++i) {
	       obj.variance_vec[i] = ptr[i];
	     }
	   })

      .def("save_sol",
	   [](StationaryCoanda &obj) {
	     obj.output_results();
	   })

      .def("get_dof_u",
	   [](StationaryCoanda &obj) {
	     return obj.get_dof_u();
	   })

      .def("get_dof_p",
	   [](StationaryCoanda &obj) {
	     return obj.get_dof_p();
	   })
      
      .def("get_coord_list",
	   [](StationaryCoanda &obj) {
	     const std::vector<Point<2>> &support_points = obj.get_support_points();
	     int num_coords = support_points.size();
	       
	     py::array_t<double, py::array::c_style | py::array::forcecast> numpy_support_points({num_coords, 2});
	     auto buffer_info = numpy_support_points.request();
	     double *ptr = static_cast<double *>(buffer_info.ptr);

	     int cunt=0;
	     for (const auto &point : support_points) {
	       cunt++;
	       for (size_t dim = 0; dim < 2; ++dim) {
		 *(ptr++) = point[dim];
	       }
	     }
	     return numpy_support_points;
	   })


      .def("run",
	   [](StationaryCoanda &obj, int max_iter, std::string solver_type, std::string strategy, double abs_tolerance, double rel_tolerance, std::string NonlinearSolver, std::string direction_Method, std::string direction_SD_ScalingType, std::string linesearch_Method,double linesearch_FS_FullStep, double linesearch_BT_MinimumStep, double linesearch_BT_DefaultStep, double linesearch_BT_RecoveryStep, double linesearch_BT_MaximumStep, double linesearch_BT_Reduction_Factor) {
	     obj.solve_system(max_iter, solver_type, strategy, abs_tolerance, rel_tolerance, NonlinearSolver, direction_Method, direction_SD_ScalingType, linesearch_Method, linesearch_FS_FullStep, linesearch_BT_MinimumStep, linesearch_BT_DefaultStep, linesearch_BT_RecoveryStep, linesearch_BT_MaximumStep, linesearch_BT_Reduction_Factor);
	   }, py::arg("max_iter")=100, py::arg("solver_type")="automatic", py::arg("strategy")="newton", py::arg("abs_tolerance")=1e-6, py::arg("rel_tolerance")=1e-15, py::arg("NonlinearSolver")="Line Search Based", py::arg("direction_Method")="Newton", py::arg("direction_SD_ScalingType")="2-Norm", py::arg("linesearch_Method")="Full Step", py::arg("linesearch_FS_FullStep")=1.0, py::arg("linesearch_BT_MinimumStep")=0.001, py::arg("linesearch_BT_DefaultStep")=1.0, py::arg("linesearch_BT_RecoveryStep")=1.0, py::arg("linesearch_BT_MaximumStep")=1.0, py::arg("linesearch_BT_Reduction_Factor")=0.5);
}

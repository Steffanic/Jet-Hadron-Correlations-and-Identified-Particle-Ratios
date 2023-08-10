import array
import pytest 

from JetHadronAnalysis.Sparse import Sparse, TriggerSparse
from JetHadronAnalysis.Types import AnalysisType
from ROOT import THnSparseD



@pytest.fixture
def empty_THnSparse_2D():
    def _create_empty_2D_sparse(nbins_x: int = 2, nbins_y: int = 2):
        return THnSparseD("test2D", "test2D", 2, array.array("i", [nbins_x, nbins_y]), array.array("d", [0, 0]), array.array("d", [10, 10]))
    return _create_empty_2D_sparse

@pytest.fixture
def empty_THnSparse_1D():
    
    return THnSparseD("test1D", "test1D", 1, array.array("i", [2]), array.array("d", [0]), array.array("d", [2]))

@pytest.fixture
def filled_THnSparse_2D(empty_THnSparse_2D):
    def _create_filled_2D_sparse(nbins_x: int = 2, nbins_y: int = 2):
        sparse = empty_THnSparse_2D(nbins_x, nbins_y)
        for i in range(100):
            sparse.Fill(array.array("d", [i%10, i%10]), 1)
        return sparse
    return _create_filled_2D_sparse

def test_add_sparse_to_sparseList(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D())
    assert sparse.getNumberOfSparses() == 1

def test_projection_without_sparses():
    sparse = Sparse(AnalysisType.PP)
    with pytest.raises(AssertionError):
        sparse.getProjection(1,2)

def test_projection_without_any_axes(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D())
    with pytest.raises(ValueError):
        sparse.getProjection()

def test_projection_along_too_many_axes(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D())
    with pytest.raises(NotImplementedError):
        sparse.getProjection(1,2,3,4,5)

def test_projection_with_single_empty_sparse(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D())
    sparse.getProjection(0)

# def test_failure_to_add_on_projection(filled_THnSparse_2D):
#     sparse = Sparse(AnalysisType.PP)
#     sparse.addSparse(filled_THnSparse_2D())
#     sparse.addSparse(filled_THnSparse_2D(nbins_x=20, nbins_y=20))
#     with pytest.raises(RuntimeError):
#         sparse.getProjection(0,1)

def test_nonzero_number_of_trigger_jets(filled_THnSparse_2D):
    sparse = TriggerSparse(AnalysisType.PP)
    sparse.addSparse(filled_THnSparse_2D())
    assert sparse.getNumberOfTriggerJets() == 100
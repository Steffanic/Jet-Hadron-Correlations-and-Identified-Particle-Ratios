import array
import pytest 

from JetHadronAnalysis.Sparse import Sparse
from JetHadronAnalysis.Types import AnalysisType
from ROOT import THnSparseD

@pytest.fixture
def empty_THnSparse_2D():
    return THnSparseD("test2D", "test2D", 2, array.array("i", [2,2]))

@pytest.fixture
def empty_THnSparse_1D():
    return THnSparseD("test1D", "test1D", 1, array.array("i", [2,2]))

def test_add_sparse_to_sparseList(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D)
    assert sparse.getNumberOfSparses() == 1

def test_projection_without_sparses():
    sparse = Sparse(AnalysisType.PP)
    with pytest.raises(AssertionError):
        sparse.getProjection(1,2)

def test_projection_without_any_axes(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D)
    with pytest.raises(ValueError):
        sparse.getProjection()

def test_projection_along_too_many_axes(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D)
    with pytest.raises(NotImplementedError):
        sparse.getProjection(1,2,3,4,5)

def test_projection_with_single_empty_sparse(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D)
    sparse.getProjection(0)

def test_failure_on_projection_with_multiple_sparses(empty_THnSparse_2D):
    sparse = Sparse(AnalysisType.PP)
    sparse.addSparse(empty_THnSparse_2D)
    sparse.addSparse(empty_THnSparse_2D)
    sparse.getProjection(0,1)
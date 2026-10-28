import numpy as np
import pandas as pd
import pytest
from mlstudio.arff_io import load_arff, export_arff
from mlstudio.data import load_csv


def test_dense_arff_roundtrip_quoted_missing_nominal_and_strings(tmp_path):
    f=pd.DataFrame({'x':[1.,np.nan,3.5], 'label':pd.Categorical(['a,b','?',None],categories=['a,b','?','unused']),
        'text':["it's % okay",'line\nbreak',None]})
    p=tmp_path/'test.arff';export_arff(f,p)
    out=load_csv(p);pd.testing.assert_frame_equal(out,f)
    assert list(out.label.cat.categories)==['a,b','?','unused']


@pytest.mark.parametrize('header,row,match',[
    ('@ATTRIBUTE x NUMERIC','{0 1}','Sparse'),
    ('@ATTRIBUTE x DATE','x','DATE'),
    ('@ATTRIBUTE x {a,b}','c','declared domain'),
    ('@ATTRIBUTE x NUMERIC','infinity','finite'),
    ('@ATTRIBUTE x NUMERIC','1,2','Expected'),
])
def test_unsupported_arff_has_actionable_error(tmp_path,header,row,match):
    p=tmp_path/'bad.arff';p.write_text('@RELATION a\n'+header+'\n@DATA\n'+row+'\n')
    with pytest.raises(ValueError,match=match):load_arff(p)


def test_failed_export_preserves_existing_file(tmp_path):
    p=tmp_path/'safe.arff';p.write_text('original')
    with pytest.raises(ValueError,match='infinity'):export_arff(pd.DataFrame({'x':[np.inf]}),p)
    assert p.read_text()=='original'

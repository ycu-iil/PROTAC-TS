import medchem as mc

from chemtsv2.filter import Filter
from chemtsv2.utils import transform_linker_to_mol

class StructuralAlertFilter(Filter):
    def check(mol, config):
        alerts = mc.structural.CommonAlertsFilters()
        try:
            results = alerts(mols=[mol],
                            n_jobs=-1,
                            progress=True,
                            progress_leave=True,
                            scheduler="auto",)
        except:
            return False
        return results["reasons"][0] == None
    
class StructuralAlertFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return StructuralAlertFilter.check(mol, conf)
        return _check(mol, conf)
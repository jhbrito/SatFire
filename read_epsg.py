from epsg_ident import EpsgIdent

ident = EpsgIdent()
ident.read_prj_from_file(r"E:\OneDrive - Instituto Politécnico do Cávado e do Ave\Desktop_backup\Tese\dados_tese\area_ardida_2017\AreasArdidas_2017_031002018_ETRS89PTTM06.prj")
print(ident.get_epsg())
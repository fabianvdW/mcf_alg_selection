cd netgen
sh ./netgen_8.sh
sh ./netgen_deg.sh
sh ./netgen_lo_8.sh
sh ./netgen_lo_sr.sh
sh ./netgen_sr.sh
for f in *.param;do ./../../../netgen/netgen.exe < $f > $f.min;done
rename .min.param.min .min *.min.param.min
rm -r *.min.param
cd ..
cd gridgen
sh ./gridgen_8.sh
sh ./gridgen_deg.sh
sh ./gridgen_sr.sh
for f in *.param;do ./../../../gridgen/gridgen.exe < $f > $f.min;done
rename .min.param.min .min *.min.param.min
rm -r *.min.param
cd ..
cd goto
sh ./goto_8.sh
sh ./goto_sr.sh
for f in *.param;do ./../../../goto/goto.exe < $f > $f.min;done
rename .min.param.min .min *.min.param.min
rm -r *.min.param
cd ..
cd gridgraph
sh ./grid_wide.sh
sh ./grid_square.sh
sh ./grid_long.sh
for f in *.param;do ./../../../gridgraph/gridgraph.exe < $f > $f.min;done
rename .min.param.min .min *.min.param.min
rm -r *.min.param
cd ..
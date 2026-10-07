# Challenge-NLA

# per compile
dalla root del progetto:

make 

# per eseguire
./main1 <image_path>


# comando lis per task 8:
mpirun -n 4 ./../../lis_test/test1 A2.mtx w.mtx x.mtx hist.txt   -i bicgstab -p ilut -tol 1e-12
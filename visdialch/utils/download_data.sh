# download ConceptNet
mkdir -p /home/$USER/KBGN-Implementation/data/cpnet/
wget -nc -P /home/$USER/KBGN-Implementation/data/cpnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd /home/$USER/KBGN-Implementation/data/cpnet/
yes n | gzip -d conceptnet-assertions-5.6.0.csv.gz
cd ../../

# download TransE embeddings
mkdir -p /home/$USER/KBGN-Implementation/data/transe/
wget -nc -P /home/$USER/KBGN-Implementation/data/transe/ https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.ent.npy
wget -nc -P /home/$USER/KBGN-Implementation/data/transe/ https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.rel.npy


# download numberbatch embeddings
wget -nc -P /home/$USER/KBGN-Implementation/data/transe/ https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
cd /home/$USER/KBGN-Implementation/data/transe/
yes n | gzip -d numberbatch-en-19.08.txt.gz
cd ../
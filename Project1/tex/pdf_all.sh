if [ -d "./out" ] 
then
	echo "Using existing auxillary directory ./out/"
else
	echo "Generating output directory ./out/"
	mkdir out
fi

for f in '*.tex'
do
	pdflatex $f --aux-directory out
done

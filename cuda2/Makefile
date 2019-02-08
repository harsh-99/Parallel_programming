NVCC = nvcc

assignment1: headers.h kernel.cu host.cu  
	$(NVCC) -I ./ kernel.cu host.cu -o assignment1

      
run:
	./assignment1

clean:
	rm -rf assignment1


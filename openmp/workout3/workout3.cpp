#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <iostream>

using namespace std;
int main(){
	int n_variable;
	printf("Enter the number of variable-:");
	cin>>n_variable;
	int i,j;
	float** var = (float**)malloc(n_variable*sizeof(float*));
	for(i=0;i<n_variable;i++){
		var[i] = (float*)malloc((n_variable+1)*sizeof(float));
	}
	cout<<"Enter the variable\n";
	for(i=0;i<n_variable;i++){
		for(j=0;j<(n_variable+1);j++){
			cin>>var[i][j];
		}
	}
	// for(i=0;i<n_variable;i++){
	// 	for(j=0;j<(n_variable+1);j++){
	// 		cout<<var[i][j]<<"  ";
	// 	}
	// 	printf("\n");
	// }

	for(int k=0;k<(n_variable-1);k++){
		#pragma omp parallel
		{
			#pragma omp for collapse(2)
				for(int i=k+1;i<(n_variable);i++){
					for(int j=(n_variable);j>=0;j--){
						var[i][j] = var[i][j] - (1.0*var[i][k]*var[k][j])/var[k][k];
				}
		}	
	}
	}
	cout<<"\n The simple form of matrix is-: \n";
	for(i=0;i<n_variable;i++){
		for(j=0;j<(n_variable+1);j++){
			cout<<var[i][j]<<"  ";
		}
		printf("\n");
	}

	float* val = (float*)malloc(n_variable*sizeof(float));
	for(int i=(n_variable-1); i>=0; i--){
		float sum = 0;
		for (int j=(n_variable-1); j>i; j--){
			sum += val[j]*var[i][j];	
		}
		val[i] = (var[i][n_variable]-sum)/var[i][i];
	}
	cout<<"The answer is :";
	for(int i=0;i<n_variable;i++){
		cout<<val[i]<<"\t";
	}
	cout<<"\n";

}
#include<iostream>
#include<fstream>
#include<math.h>
#include<vector>
#include<stdlib.h>
#include<omp.h>

#define n_balls 1000
#define dim_x 100
#define dim_y 200
#define dim_z 400
#define del_t 0.01
#define threads 1
#define NUM_STEPS 72000

using namespace std;

class Ball
{
public:
	int index;
	float x, y, z;
	float fx, fy, fz;
	float vx, vy, vz;
public:
	void initialize(float, float, float, int);
	void calc_force(Ball*);
	void update_vel_half();
	void update_pos();
	void update_vel_full();
	// void check_collision();
	void test();
};

void Ball::initialize(float a, float b, float c, int ind)
{
	fx = fy = fx = vx = vy = vz = 0.0;
	x = a;
	y = b;
	z = c;
	index = ind;
}
void Ball::test()
{
	cout<<vx<<' '<<vy<<' '<<vz<<endl;
}

void Ball::calc_force(Ball *B)
{
	#pragma omp parallel for num_threads(threads) shared(B)
	for(int i = 0; i < n_balls; i++)
	{
		if(i == index)
			continue;
		fx += (B[i].x - x)/pow(pow(B[i].x - x, 2) + pow(B[i].y - y, 2) + pow(B[i].z - z, 2), 1.5);
		fy += (B[i].y - y)/pow(pow(B[i].x - x, 2) + pow(B[i].y - y, 2) + pow(B[i].z - z, 2), 1.5);
		fz += (B[i].z - z)/pow(pow(B[i].x - x, 2) + pow(B[i].y - y, 2) + pow(B[i].z - z, 2), 1.5);
	}
}

void Ball::update_vel_half()
{
	vx = vx + fx*del_t/2;
	vy = vy + fy*del_t/2;
	vz = vz + fz*del_t/2;
}

void Ball::update_pos()
{
	x = x + vx*del_t;
	if(x > dim_x)
	{
		x = dim_x - (x - dim_x);
		vx = -1*vx;
	}
	if(x < 0)
	{
		x = abs(x);
		vx = -1*vx;
	}

	y = y + vy*del_t;
	if(y >dim_y)
	{
		y = dim_y - (y - dim_y);
		vy = -1*vy;
	}
	if (y < 0)
	{
		y = abs(y);
		vy = -1*vy;
	}

	z = z + vz*del_t;
	if(z >dim_z)
	{
		z = dim_z - (z - dim_z);
		vz = -1*vz;
	}
	if (z < 0)
	{
		z = abs(z);
		vz = -1*vz;
	}	

}

void Ball::update_vel_full()
{
	vx = vx + fx*del_t/2;
	vy = vy + fy*del_t/2;
	vz = vz + fz*del_t/2;
}

void check_collision(Ball* B)
{
	vector<int> ***Grid = new vector<int>**[11];
	for(int i = 0; i < 11; i++)
	{
		Grid[i] = new vector<int>*[21];
		for(int j = 0; j < 21; j++)
			Grid[i][j] = new vector<int>[41];
	}
	for(int i = 0; i < 1000; i++)
	{
		// cout<<(int)round(B[i].x/10)<<' '<<(int)round(B[i].y/10)<<' '<<(int)round(B[i].z/10)<<endl;
		Grid[(int)round(B[i].x/10)][(int)round(B[i].y/10)][(int)round(B[i].z/10)].push_back(i);

	}
	#pragma omp parallel num_threads(threads) shared(B, Grid) 
	{
		#pragma omp for collapse(3)
		for(int i = 0; i < 11; i++)
		{
			for(int j = 0; j < 21; j++)
			{
				for(int k = 0; k < 41; k++)
				{
					if(Grid[i][j][k].size() < 2)
						continue;
					for(int m = 0; m < Grid[i][j][k].size(); m++)
					{
						for(int n = m + 1; n < Grid[i][j][k].size(); n++)
						{
							if(pow(B[Grid[i][j][k][m]].x - B[Grid[i][j][k][n]].x, 2) + pow(B[Grid[i][j][k][m]].y - B[Grid[i][j][k][n]].y, 2) + pow(B[Grid[i][j][k][m]].z - B[Grid[i][j][k][n]].z, 2) > 1.01)
								continue;
							float temp = B[Grid[i][j][k][m]].vx;
							B[Grid[i][j][k][m]].vx = B[Grid[i][j][k][n]].vx;
							B[Grid[i][j][k][n]].vx = temp;

							temp = B[Grid[i][j][k][m]].vy;
							B[Grid[i][j][k][m]].vy = B[Grid[i][j][k][n]].vy;
							B[Grid[i][j][k][n]].vy = temp;

							temp = B[Grid[i][j][k][m]].vz;
							B[Grid[i][j][k][m]].vz = B[Grid[i][j][k][n]].vz;
							B[Grid[i][j][k][n]].vz = temp;
						}
					}
				}
			}
		}
	}
}

int main()
{
	ifstream fin("Trajectory.txt", ios::in);
	string str;
	for(int i = 0; i < 8; i++)
		getline(fin, str);
	float a, b, c;
	Ball *B = new Ball[n_balls];

	//This cannot be parallelized as reading of file is a serial 
	for(int i = 0; i < n_balls; i++)
	{
		fin>>a>>b>>c;
		B[i].initialize(a, b, c, i);
	}

	ofstream fout("out.txt");

	double tot_time = 0;

	for(int step = 0; step < NUM_STEPS; step++)
	{
		fout<<"Step : "<<step<<endl;
		double wtime = omp_get_wtime();

		#pragma omp parallel for num_threads(threads) shared(B)
		for(int i = 0; i < n_balls; i++)
		{
			B[i].calc_force(B);
			B[i].update_vel_half();
			B[i].update_pos();
			B[i].update_vel_full();
		}

		check_collision(B);
		wtime = omp_get_wtime() - wtime;
		tot_time += wtime; 
		if(step % 100 == 0)
		{
			for(int i = 0; i < n_balls; i++)
				fout<<B[i].x<<' '<<B[i].y<<' '<<B[i].z<<' ';
			fout<<endl;
		}
		cout<<"Step : "<<step<<" Time : "<<wtime<<" Total Time: "<<tot_time<<endl;
	}
	
	fout.close();
}

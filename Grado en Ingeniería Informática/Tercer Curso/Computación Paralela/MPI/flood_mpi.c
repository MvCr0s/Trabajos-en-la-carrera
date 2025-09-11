/*
 * Simulation of rainwater flooding
 *
 * Reference sequential version (Do not modify this code)
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 *
 * v1.6
 *
 * (c) 2025 Arturo Gonzalez-Escribano and Diego Garcia Alvarez
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <mpi.h>

/*
 * Water levels are stored with fixed preciseon
 * This avoids result differences when arithmetic operations are reordered
 */
#define PRECISION 1000000
#define FIXED(a) ((int)((a) * PRECISION))
#define FLOATING(a) ((float)(a) / PRECISION)
#define PRECISION_FIXED 1
#define PRECISION_FLOAT 2

/*
 * Scenario size (km x km)
 */
#define SCENARIO_SIZE 30

/*
 * Spillage factor for equilibrium
 */
#define SPILLAGE_FACTOR 2

/*
 * Utils: Random generator
 */
#include "rng.c"

/*
 * Utils: Function to get wall time
 */
double cp_Wtime()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

/*
 * Utils: Number of contiguous cells to consider for water spillage
 * 	0: up, 1: down, 2: left, 3: right
 * 	Displacements for the contiguous cells
 * 	This data structure can be changed and/or optimized by the students
 */
#define CONTIGUOUS_CELLS 4
int displacements[CONTIGUOUS_CELLS][2] = {
	{-1, 0}, // Top
	{1, 0},	 // Bottom
	{0, -1}, // Left
	{0, 1}	 // Right
};

/*
 * Utils: Macro-functions to transform coordinates, from scenario to matrix cells, and back
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define SCENARIO_SIZE 30
#define COORD_SCEN2MAT_X(x) (x * columns / SCENARIO_SIZE)
#define COORD_SCEN2MAT_Y(y) (y * rows / SCENARIO_SIZE)
#define COORD_MAT2SCEN_X(c) (c * SCENARIO_SIZE / columns)
#define COORD_MAT2SCEN_Y(r) (r * SCENARIO_SIZE / rows)

/*
 * Utils: Macro functions for the min and max of two numbers
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
 * Utils: Macro function to simplify accessing data of 2D and 3D matrixes stored in a flattened array
 * 	These macro-functions can be changed and/or optimized by the students
 */
#define accessMat(arr, i, j) (arr[((int)(i) - my_start_row + 1) * columns + (int)(j)])
#define accessMat3D(arr, i, j, k) (arr[(((int)(i) - my_start_row + 1) * columns * CONTIGUOUS_CELLS) + ((int)(j) * CONTIGUOUS_CELLS) + (int)(k)])

#define accessMatNoHalo(arr, i, j) ((arr)[((int)(i) - my_start_row) * columns + (int)(j)])


/*
 * Function: Generate ground height for a given position
 * 	This function can be changed and/or optimized by the students
 */
float get_height(char scenario, int row, int col, int rows, int columns)
{
	// Choose scenario limits
	float x_min, x_max, y_min, y_max;
	if (scenario == 'M')
	{ // Mountains scenario
		x_min = -3.3;
		x_max = 5.1;
		y_min = -0.5;
		y_max = 8.8;
	}
	else
	{ // Valley scenarios
		x_min = -5.5;
		x_max = -3;
		y_min = -0.1;
		y_max = 4.2;
	}

	// Compute scenario coordinates of the cell position
	float x = x_min + ((x_max - x_min) / columns) * col;
	float y = y_min + ((y_max - y_min) / rows) * row;

	// Compute function height
	float height = -1 / (x * x + 1) + 2 / (y * y + 1) + 0.5 * sin(5 * sqrt(x * x + y * y)) / sqrt(x * x + y * y) + (x + y) / 3 + sin(x) * cos(y) + 0.4 * sin(3 * x + y) + 0.25 * cos(4 * y + x);

// Substitute by the dam height in the proper scenarios
#define LOW_DAM_HEIGHT -1.0
#define HIGH_DAM_HEIGHT -0.4
	if (scenario == 'D' && x <= -4.96 && x >= -5.0)
	{
		if (height < HIGH_DAM_HEIGHT)
			height = HIGH_DAM_HEIGHT;
	}
	else if (scenario == 'd' && x <= -5.3 && x >= -5.34)
	{
		if (height < LOW_DAM_HEIGHT)
			height = LOW_DAM_HEIGHT;
	}

	// Transform to meters
	if (scenario == 'M')
		return height * 30 + 400;
	else
		return height * 20 + 100;
}

/*
 * Structure to represent moving rainy clouds
 * 	This structure can be changed and/or optimized by the students
 */
typedef struct
{
	float x;		 // x coordinate of the center
	float y;		 // y coordinate of the center
	float radius;	 // radius of the cloud (km)
	float intensity; // rainfall intensity (cm/h)
	float speed;	 // speed of movement (km/h)
	float angle;	 // angle of movement
	int active;		 // active cloud
} Cloud_t;

/*
 * Function: Initialize cloud with random values
 * 	This function can be changed and/or optimized by the students
 */
Cloud_t cloud_init(Cloud_t cloud_model, float front_distance, float front_width, float front_depth, float front_direction, int rows, int cols, rng_t *rnd_state)
{
	Cloud_t cloud;

	// Random position around the front center
	cloud.x = (float)rng_next_between(rnd_state, 0, front_width) - front_width / 2;
	cloud.y = (float)rng_next_between(rnd_state, 0, front_depth) - front_depth / 2;

	// Rotate
	float opposite = front_direction + 180;
	float tmp_x = cloud.x;
	float tmp_y = cloud.y;
	cloud.x = tmp_x * cos(opposite * M_PI / 180.0) - tmp_y * sin(opposite * M_PI / 180.0);
	cloud.y = tmp_x * sin(opposite * M_PI / 180.0) + tmp_y * cos(opposite * M_PI / 180.0);

	// Move center
	float x_center = front_distance * cos(opposite * M_PI / 180.0) + SCENARIO_SIZE / 2;
	float y_center = front_distance * sin(opposite * M_PI / 180.0) + SCENARIO_SIZE / 2;
	cloud.x += x_center;
	cloud.y += y_center;

	// Cloud random parameters
	cloud.radius = (float)rng_next_between(rnd_state, cloud_model.radius / 2, cloud_model.radius);
	cloud.intensity = (float)rng_next_between(rnd_state, cloud_model.intensity / 2, cloud_model.intensity);
	cloud.speed = (float)rng_next_between(rnd_state, cloud_model.speed / 2, cloud_model.speed);
	cloud.angle = front_direction + (float)rng_next_between(rnd_state, 0, cloud_model.angle) - cloud_model.angle / 2;
	cloud.active = 1;
	return cloud;
}

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

#ifdef DEBUG
/*
 * Function: Print the current state of the simulation
 */
void print_matrix(int precision_type, int rows, int columns, void *mat, const char *msj)
{
	/*
	 * You don't need to optimize this function, it is only for pretty
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i, j;
#ifndef ANIMATION
	printf("%s:\n", msj);
	printf("+");
	for (j = 0; j < columns; j++)
		printf("----------");
	printf("+\n");
	printf("\n");
#endif
	// Y coordinates: Reversed, higher rows up
	for (i = rows - 1; i >= 0; i--)
	{
#ifndef ANIMATION
		printf("|");
#endif
		// X coordinates
		for (j = 0; j < columns; j++)
		{
			if (precision_type == PRECISION_FLOAT)
				printf(" %10.4f", accessMat(((float *)mat), i, j));
			else
				printf(" %10.4f", FLOATING(accessMat(((int *)mat), i, j)));
		}
#ifndef ANIMATION
		printf("|\n");
#endif
		printf("\n");
	}
#ifndef ANIMATION
	printf("+");
	for (j = 0; j < columns; j++)
		printf("----------");
	printf("+\n\n\n");
#else
	printf("\n");
#endif
}

/*
 * Function: Print the current state of the clouds
 */
void print_clouds(int num_clouds, Cloud_t *clouds)
{
	/*
	 * You don't need to optimize this function, it is only for pretty
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	printf("Clouds:\n");
	for (int i = 0; i < num_clouds; i++)
	{
		printf("Cloud %d: x = %f, y = %f, radius = %f, intensity = %f, speed = %f, angle = %f\n",
			   i,
			   clouds[i].x,
			   clouds[i].y,
			   clouds[i].radius,
			   clouds[i].intensity,
			   clouds[i].speed,
			   clouds[i].angle);
	}
	printf("\n");
}
#endif // DEBUG

/*
 * Function: Print the program usage line in stderr
 */
void show_usage(char *program_name)
{
	fprintf(stderr, "\nFlood Simulation - Simulate rain and flooding in %d x %d km^2\n", SCENARIO_SIZE, SCENARIO_SIZE);
	fprintf(stderr, "----------------------------------------------------------------\n");
	fprintf(stderr, "Usage: %s ", program_name);
	fprintf(stderr, "<rows> <columns> <ground_scenario(M|V|D|d)> <threshold> <num_minutes> <exaggeration_factor> <front_distance> <front_width> <front_depth> <front_direction(grad.)> <num_random_clouds> <cloud_max_radius(km)> <cloud_max_intensity(mm/h)> <cloud_max_speed(km/h)> <cloud_max_angle_aperture(grad.)> <clouds_rnd_seed>\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "\tOptional arguments for special clouds: <cloud_start_x(km)> <cloud_start_y(km)> <cloud_radius(km)> <cloud_intensity(mm/h)> <cloud_speed(km/h)> <cloud_angle(grad.)> ...\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "\tGround models: 'M' mountain lakes, 'V' valley, 'D' valley with high dam, 'd' valley with low dam\n");
	fprintf(stderr, "\tIntensity of rain (mm/h): Strong (15-30), Very strong (30-60), Torrential: Above 60\n");
	fprintf(stderr, "\n");
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[])
{
#ifdef DEBUG
	setbuf(stdout, NULL);
	setbuf(stderr, NULL);
#endif

	int mpi_rank2, mpi_nprocs2;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank2);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_nprocs2);

#define NUM_FIXED_ARGS 17

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < NUM_FIXED_ARGS)
	{
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage(argv[0]);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	if (argc > NUM_FIXED_ARGS)
	{
		if ((argc - NUM_FIXED_ARGS) % 6 != 0)
		{
			fprintf(stderr, "-- Error: Wrong number of arguments, there should be %d compulsory arguments + groups of 6 optional arguments\n", NUM_FIXED_ARGS);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
	}

	/* 1.2. Read ground sizes and selection of ground scenario */
	int rows = atoi(argv[1]);
	int columns = atoi(argv[2]);
	char ground_scenario = argv[3][0];
	if (ground_scenario != 'M' && ground_scenario != 'V' && ground_scenario != 'D' && ground_scenario != 'd')
	{
		fprintf(stderr, "-- Error: Wrong ground scenario\n\n");
		show_usage(argv[0]);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	/* 1.3. Read termination conditions */
	float threshold = atof(argv[4]);
	int num_minutes = atoi(argv[5]);

	/* 1.4. Read clouds data */
	float ex_factor = atoi(argv[6]);
	float front_distance = atof(argv[7]);
	float front_width = atof(argv[8]);
	float front_depth = atof(argv[9]);
	float front_direction = atof(argv[10]);
	int num_clouds = atoi(argv[11]);
	Cloud_t cloud_model;
	cloud_model.x = cloud_model.y = 0;
	cloud_model.radius = atof(argv[12]);
	cloud_model.intensity = atof(argv[13]);
	cloud_model.speed = atof(argv[14]);
	cloud_model.angle = atof(argv[15]);
	cloud_model.active = 0;
	unsigned int seed_clouds = (unsigned int)atol(argv[16]);
	// Initialize random sequence
	rng_t rnd_state = rng_new(seed_clouds);

	/* 1.5. Read the non-random clouds information */
	int num_clouds_arg = (argc - NUM_FIXED_ARGS) / 6;
	Cloud_t clouds_arg[num_clouds_arg];
	int idx;
	for (idx = NUM_FIXED_ARGS; idx < argc; idx += 6)
	{
		int pos = (idx - NUM_FIXED_ARGS) / 6;
		clouds_arg[pos].x = atof(argv[idx]);
		clouds_arg[pos].y = atof(argv[idx + 1]);
		clouds_arg[pos].radius = atof(argv[idx + 2]);
		clouds_arg[pos].intensity = atof(argv[idx + 3]);
		clouds_arg[pos].speed = atof(argv[idx + 4]);
		clouds_arg[pos].angle = atof(argv[idx + 5]);
	}

#ifdef DEBUG
#ifdef ANIMATION
	printf("%d %d\n", rows, columns);
	printf("%d\n", num_minutes + 1);
#else
	if (mpi_rank2 == 0)
	{
		/* 1.5. Print arguments */
		printf("Arguments, Num_minutes: %d\n", num_minutes);
		printf("Arguments, Rows: %d, Columns: %d\n", rows, columns);
		printf("Arguments, Groud scenario: %c\n", ground_scenario);
		printf("Arguments, Num_clouds: %d, Max_radius: %f, Max_intensity: %f, Max_speed: %f, Max_angle: %f, seed: %u\n",
			   num_clouds,
			   cloud_model.radius,
			   cloud_model.intensity,
			   cloud_model.speed,
			   cloud_model.angle,
			   seed_clouds);
		for (idx = 0; idx < num_clouds_arg; idx++)
		{
			printf("Arguments, Optional cloud %d: x: %f, y: %f, Radius: %f, Intensity: %f, Speed: %f, Angle: %f\n",
				   idx,
				   clouds_arg[idx].x,
				   clouds_arg[idx].y,
				   clouds_arg[idx].radius,
				   clouds_arg[idx].intensity,
				   clouds_arg[idx].speed,
				   clouds_arg[idx].angle);
		}
		printf("\n");
	}
#endif
#endif

	/* 2. Start global timer */
	MPI_Barrier(MPI_COMM_WORLD);
	double ttotal = cp_Wtime();

	/*
	 *
	 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
	 *
	 */

	int *water_level = NULL;		   // Level of water on each cell (fixed precision)
	float *ground = NULL;			   // Ground height
	float *spillage_flag = NULL;	   // Indicates which cells are spilling to neighbors
	float *spillage_level = NULL;	   // Maximum level of spillage of each cell
	float *spillage_from_neigh = NULL; // Spillage from each neighbor
	Cloud_t *clouds = NULL;			   // Clouds

	// float *up_g = NULL;
	// int *up_w = NULL;
	// float *down_g = NULL;
	// int *down_w = NULL;

	float *topHaloSend = NULL;
	float *topHaloRecv = NULL;
	float *bottomHaloSend = NULL;
	float *bottomHaloRecv = NULL;

	/* Calcular distribución de filas por proceso */
	int rows_per_process = rows / mpi_nprocs2;
	int remaining_rows = rows % mpi_nprocs2;

	int my_rows = rows_per_process + (mpi_rank2 < remaining_rows ? 1 : 0);
	int my_start_row = mpi_rank2 * rows_per_process + (mpi_rank2 < remaining_rows ? mpi_rank2 : remaining_rows);
	int my_end_row = my_start_row + my_rows;

	/* Crear comunicador solo para procesos activos */
	MPI_Comm active_comm;
	int active = (my_rows > 0) ? 1 : MPI_UNDEFINED;
	MPI_Comm_split(MPI_COMM_WORLD, active, mpi_rank2, &active_comm);

	/* Si no tienes filas, sales */
	if (my_rows == 0)
	{
		MPI_Finalize();
		return 0;
	}

	int mpi_rank, mpi_nprocs;
	MPI_Comm_rank(active_comm, &mpi_rank);
	MPI_Comm_size(active_comm, &mpi_nprocs);

	/* 3. Initialization */
	/* 3.1. Memory allocation */
	// up_g = (float *)malloc(columns * sizeof(float));
	// up_w = (int *)malloc(columns * sizeof(int));
	// down_g = (float *)malloc(columns * sizeof(float));
	// down_w = (int *)malloc(columns * sizeof(int));

	topHaloSend = (float *)malloc(columns * sizeof(float));
	topHaloRecv = (float *)malloc(columns * sizeof(float));
	bottomHaloSend = (float *)malloc(columns * sizeof(float));
	bottomHaloRecv = (float *)malloc(columns * sizeof(float));

	ground = malloc(sizeof(float) * (my_rows + 2) * columns);
	water_level = calloc((my_rows + 2) * columns, sizeof(int));
	spillage_flag = calloc(my_rows * columns, sizeof(float));
	spillage_level = calloc(my_rows * columns, sizeof(float));
	spillage_from_neigh = calloc((my_rows + 2) * columns * CONTIGUOUS_CELLS, sizeof(float));

	int total_clouds = num_clouds + num_clouds_arg;


	clouds = (Cloud_t *)calloc(total_clouds, sizeof(Cloud_t));
	if (clouds == NULL)
	{
		fprintf(stderr, "-- Error allocating clouds structures for total size: %d\n", total_clouds);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	if (ground == NULL || water_level == NULL || spillage_flag == NULL || spillage_level == NULL || spillage_from_neigh == NULL)
	{
		fprintf(stderr, "-- Error allocating ground and rain structures for size: %d x %d \n", rows, columns);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	/* 3.2. Ground generation */
	int row_pos, col_pos;
	for (row_pos = my_start_row - 1; row_pos < my_end_row + 1; row_pos++)
	{
		if (!((mpi_rank == 0 && row_pos == -1) || (mpi_rank == mpi_nprocs - 1 && row_pos == my_rows + 1)))
		{
			for (col_pos = 0; col_pos < columns; col_pos++)
				accessMat(ground, row_pos, col_pos) = get_height(ground_scenario, row_pos, col_pos, rows, columns);
		}
	}

#ifdef DEBUG
	print_matrix(PRECISION_FLOAT, rows, columns, ground, "Ground heights");
#endif

	/* 3.3. Clouds initialization */
	/* 3.3.1 Random clouds generation */
	int cloud;
	for (cloud = 0; cloud < num_clouds; cloud++)
	{
		clouds[cloud] = cloud_init(cloud_model, front_distance, front_width, front_depth, front_direction, rows, columns, &rnd_state);
	}
	/* 3.3.2 Copy optional argument clouds */
	for (cloud = 0; cloud < num_clouds_arg; cloud++)
	{
		clouds[num_clouds + cloud] = clouds_arg[cloud];
	}

	num_clouds += num_clouds_arg;

#ifdef DEBUG
	print_matrix(PRECISION_FLOAT, rows, columns, ground, "Ground heights");
#endif

	/* Initialize metrics */
	float max_water_scenario = 0.0;
	double vertido = 0;
	double max_spillage_iter = DBL_MAX;
	double global_max_spillage = 0;
	int max_spillage_minute = 0;
	double max_spillage_scenario = 0.0;
	long total_water = 0;
	long total_water_loss = 0;
	long total_rain = 0;

	/* 4. Flood simulation (time iterations) */
	int minute;
	for (minute = 0; minute < num_minutes && max_spillage_iter > threshold; minute++)
	{
		int new_row, new_col;
		int cell_pos;

		/* 4.1. Clouds movement */
		for (cloud = 0; cloud < num_clouds; cloud++)
		{
			// Calculate new position (x are rows, y are columns)
			float km_minute = clouds[cloud].speed / 60;
			clouds[cloud].x += km_minute * cos(clouds[cloud].angle * M_PI / 180.0);
			clouds[cloud].y += km_minute * sin(clouds[cloud].angle * M_PI / 180.0);
		}

#ifdef DEBUG
#ifndef ANIMATION
		if (mpi_rank == 0)
		{
			print_clouds(num_clouds, clouds);
		}
#endif
#endif

		/* 4.2. Rainfall */
		for (cloud = 0; cloud < num_clouds; cloud++)
		{
			// Compute the bounding box area of the cloud
			float row_start = COORD_SCEN2MAT_Y(MAX(0, clouds[cloud].y - clouds[cloud].radius));
			float row_end = COORD_SCEN2MAT_Y(MIN(clouds[cloud].y + clouds[cloud].radius, SCENARIO_SIZE));
			float col_start = COORD_SCEN2MAT_X(MAX(0, clouds[cloud].x - clouds[cloud].radius));
			float col_end = COORD_SCEN2MAT_X(MIN(clouds[cloud].x + clouds[cloud].radius, SCENARIO_SIZE));
			float distance;

			// Add rain to the ground water level
			float row_pos, col_pos;
			for (row_pos = row_start; row_pos < row_end; row_pos++)
			{
				if (row_pos < my_end_row && row_pos >= my_start_row)
				{
					for (col_pos = col_start; col_pos < col_end; col_pos++)
					{
						float x_pos = COORD_MAT2SCEN_X(col_pos);
						float y_pos = COORD_MAT2SCEN_Y(row_pos);
						distance = sqrt(pow(x_pos - clouds[cloud].x, 2) + pow(y_pos - clouds[cloud].y, 2));
						if (distance < clouds[cloud].radius)
						{
							float rain = ex_factor * MAX(0, clouds[cloud].intensity - distance / clouds[cloud].radius * sqrt(clouds[cloud].intensity));
							float meters_per_minute = rain / 1000 / 60;
							accessMat(water_level, row_pos, col_pos) += FIXED(meters_per_minute);
							total_rain += FIXED(meters_per_minute);
						}
					}
				}
			}
		}

#ifdef DEBUG
		print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after rain");
#endif

		// Intercambio de halos con MPI_Sendrecv usando arrays externos
		MPI_Request request[2];

		// WATER LEVEL - fila superior (recv en up_w, desde fila final del anterior)
		if (mpi_rank > 0)
		{
			/*MPI_Sendrecv(&accessMat(water_level, my_start_row, 0), columns, MPI_INT, mpi_rank - 1, 0,
						 up_w, columns, MPI_INT, mpi_rank - 1, 2,
						 active_comm, MPI_STATUS_IGNORE);*/

			MPI_Isend(&accessMat(water_level, my_start_row, 0), columns, MPI_INT, mpi_rank - 1, 0,
					  active_comm, &request[0]);
			MPI_Recv(&accessMat(water_level, my_start_row - 1, 0), columns, MPI_INT, mpi_rank - 1, 0,
					 active_comm, MPI_STATUS_IGNORE);
		}

		// WATER LEVEL - fila inferior (recv en down_w, desde fila inicial del siguiente)
		if (mpi_rank < mpi_nprocs - 1)
		{
			/*MPI_Sendrecv(&accessMat(water_level, my_end_row - 1, 0), columns, MPI_INT, mpi_rank + 1, 2,
						 down_w, columns, MPI_INT, mpi_rank + 1, 0,
						 active_comm, MPI_STATUS_IGNORE);*/
			MPI_Isend(&accessMat(water_level, my_end_row - 1, 0), columns, MPI_INT, mpi_rank + 1, 0,
					  active_comm, &request[1]);
			MPI_Recv(&accessMat(water_level, my_end_row, 0), columns, MPI_INT, mpi_rank + 1, 0,
					 active_comm, MPI_STATUS_IGNORE);
		}

		// GROUND - fila superior (solo en minuto 0)
		/*if (minute == 0 && mpi_rank > 0) {
			MPI_Sendrecv(&accessMat(ground, my_start_row, 0), columns, MPI_FLOAT, mpi_rank - 1, 1,
						 up_g, columns, MPI_FLOAT, mpi_rank - 1, 3,
						 active_comm, MPI_STATUS_IGNORE);
		}*/

		if (mpi_rank > 0)
		{
			MPI_Wait(&request[0], MPI_STATUSES_IGNORE);
		}
		if (mpi_rank < mpi_nprocs - 1)
		{
			MPI_Wait(&request[1], MPI_STATUSES_IGNORE);
		}

		// GROUND - fila inferior (solo en minuto 0)
		/*if (minute == 0 && mpi_rank < mpi_nprocs - 1) {
			MPI_Sendrecv(&accessMat(ground, my_end_row - 1, 0), columns, MPI_FLOAT, mpi_rank + 1, 3,
						 down_g, columns, MPI_FLOAT, mpi_rank + 1, 1,
						 active_comm, MPI_STATUS_IGNORE);
		}*/

		// Copiar arrays recibidos en las filas halo
		/*if (mpi_rank > 0) {
			for (int i = 0; i < columns; i++) {
				//accessMat(water_level, my_start_row - 1, i) = up_w[i];
				if (minute == 0)
					accessMat(ground, my_start_row - 1, i) = up_g[i];
			}
		}*/

		/*if (mpi_rank < mpi_nprocs - 1) {
			for (int i = 0; i < columns; i++) {
				//accessMat(water_level, my_end_row, i) = down_w[i];
				if (minute == 0) {
					accessMat(ground, my_end_row, i) = down_g[i];
				}
			}
		}*/

		/* 4.3. Compute water spillage to neighbor cells */
		for (row_pos = my_start_row; row_pos < my_end_row; row_pos++)
		{
			for (col_pos = 0; col_pos < columns; col_pos++)
			{
				if (accessMat(water_level, row_pos, col_pos) > 0)
				{
					float sum_diff = 0;
					float my_spillage_level = 0;

					/* 4.3.1. Differences between current-cell level and its neighbours  */

					float current_height = accessMat(ground, row_pos, col_pos) + FLOATING(accessMat(water_level, row_pos, col_pos));

					// Iterate over the four neighboring cells using the displacement array
					for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++)
					{
						new_row = row_pos + displacements[cell_pos][0];
						new_col = col_pos + displacements[cell_pos][1];

						float neighbor_height;

						if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns)
							// Out of borders: Same height as the cell with no water
							neighbor_height = accessMat(ground, row_pos, col_pos);

						else
							// Neighbor cell: Ground height + water level
							neighbor_height = accessMat(ground, new_row, new_col) + FLOATING(accessMat(water_level, new_row, new_col));

						// Compute level differences
						if (current_height >= neighbor_height)
						{
							float height_diff = current_height - neighbor_height;
							sum_diff += height_diff;
							my_spillage_level = MAX(my_spillage_level, height_diff);
						}
					}
					my_spillage_level = MIN(FLOATING(accessMat(water_level, row_pos, col_pos)), my_spillage_level);

					// Compute proportion of spillage to each neighbor
					if (sum_diff > 0.0)
					{
						float proportion = my_spillage_level / sum_diff;
						// If proportion is significative, spillage
						if (proportion > 1e-8)
						{
							accessMatNoHalo(spillage_flag, row_pos, col_pos) = 1;
							accessMatNoHalo(spillage_level, row_pos, col_pos) = my_spillage_level;

							// Iterate over the four neighboring cells using the displacement array
							for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++)
							{
								new_row = row_pos + displacements[cell_pos][0];
								new_col = col_pos + displacements[cell_pos][1];

								float neighbor_height;

								// Check if the new position is within the matrix boundaries
								if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns)
								{
									// Spillage out of the borders: Water loss
									neighbor_height = accessMat(ground, row_pos, col_pos);
									if (current_height >= neighbor_height)
									{
										// esto no es
										float spilled = proportion * (current_height - neighbor_height) / 2;
										// printf("Proc %d, minute %d, LOST at (%d,%d): %.6f\n", mpi_rank, minute, row_pos, col_pos, value);
										total_water_loss += FIXED(spilled);
									}
								}
								else
								{
									neighbor_height = accessMat(ground, new_row, new_col) + FLOATING(accessMat(water_level, new_row, new_col));
									if (current_height >= neighbor_height)
									{
										float height_diff = current_height - neighbor_height;
										float value = proportion * height_diff;
										accessMat3D(spillage_from_neigh, new_row, new_col, cell_pos) = value;
									}
								}
							}
						}
					}
				}
			}
		}

		// Preparar halos a enviar
		for (col_pos = 0; col_pos < columns; col_pos++)
		{
			// Enviar fila inferior (abajo) desde nuestra última fila real
			if (my_end_row < rows)
				bottomHaloSend[col_pos] = accessMat3D(spillage_from_neigh, my_end_row, col_pos, 1); // hacia abajo

			// Enviar fila superior (arriba) desde la fila anterior a la primera real
			if (my_start_row > 0)
			{
				topHaloSend[col_pos] = accessMat3D(spillage_from_neigh, my_start_row - 1, col_pos, 0); // hacia arriba
			}
		}

		// Comunicación con procesos vecinos usando Sendrecv
		if (mpi_rank > 0)
		{
			MPI_Sendrecv(topHaloSend, columns, MPI_FLOAT, mpi_rank - 1, 0,
						 bottomHaloRecv, columns, MPI_FLOAT, mpi_rank - 1, 1,
						 active_comm, MPI_STATUS_IGNORE);
		}

		if (mpi_rank < mpi_nprocs - 1)
		{
			MPI_Sendrecv(bottomHaloSend, columns, MPI_FLOAT, mpi_rank + 1, 1,
						 topHaloRecv, columns, MPI_FLOAT, mpi_rank + 1, 0,
						 active_comm, MPI_STATUS_IGNORE);
		}

		// Copiar halos recibidos en las filas reales correspondientes
		if (mpi_rank > 0)
			for (col_pos = 0; col_pos < columns; col_pos++)
				accessMat3D(spillage_from_neigh, my_start_row, col_pos, 1) = bottomHaloRecv[col_pos]; // lo que viene de abajo

		if (mpi_rank < mpi_nprocs - 1)
			for (col_pos = 0; col_pos < columns; col_pos++)
				accessMat3D(spillage_from_neigh, my_end_row - 1, col_pos, 0) = topHaloRecv[col_pos]; // lo que viene de arriba

		/* 4.5. Propagation of previuosly computer water spillage to/from neighbors */
		max_spillage_iter = 0.0;
		for (row_pos = my_start_row; row_pos < my_end_row; row_pos++)
		{
			for (col_pos = 0; col_pos < columns; col_pos++)
			{
				// If the cell has spillage
				if (accessMatNoHalo(spillage_flag, row_pos, col_pos) == 1)
				{
					// Eliminate the spillage from the origin cell
					float spillage = accessMatNoHalo(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR;
					accessMat(water_level, row_pos, col_pos) -= FIXED(spillage);

					// Compute termination condition: Maximum cell spillage during the iteration
					if (spillage > max_spillage_iter)
						max_spillage_iter = spillage;

					// Statistics: Record maximum cell spillage during the scenario and its time
					if (spillage > max_spillage_scenario)
					{
						max_spillage_scenario = spillage;
						max_spillage_minute = minute;
					}
				}

				// Accumulate spillage from neighbors
				for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++)
					accessMat(water_level, row_pos, col_pos) += FIXED(accessMat3D(spillage_from_neigh, row_pos, col_pos, cell_pos) / SPILLAGE_FACTOR);
			}
		}

		vertido = 0;
		MPI_Allreduce(&max_spillage_iter, &vertido, 1, MPI_DOUBLE, MPI_MAX, active_comm);
		max_spillage_iter = vertido;

		MPI_Allreduce(&max_spillage_scenario, &global_max_spillage, 1, MPI_DOUBLE, MPI_MAX, active_comm);

		if (max_spillage_scenario < global_max_spillage)
		{
			max_spillage_scenario = global_max_spillage;
			max_spillage_minute = minute;
		}

#ifdef DEBUG
#ifndef ANIMATION
		print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
#endif
#endif

		/* 4.6. Reset ancillary structures */
		for (row_pos = my_start_row; row_pos < my_end_row; row_pos++)
		{
			for (col_pos = 0; col_pos < columns; col_pos++)
			{
				for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++)
				{
					accessMat3D(spillage_from_neigh, row_pos, col_pos, cell_pos) = 0;
				}
				accessMatNoHalo(spillage_flag, row_pos, col_pos) = 0;
				accessMatNoHalo(spillage_level, row_pos, col_pos) = 0;
			}
		}

		for (col_pos = 0; col_pos < columns; col_pos++)
		{
			for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++)
			{
				if (mpi_rank > 0)
					accessMat3D(spillage_from_neigh, my_start_row - 1, col_pos, cell_pos) = 0;
				if (mpi_rank < mpi_nprocs - 1)
					accessMat3D(spillage_from_neigh, my_end_row, col_pos, cell_pos) = 0;
			}
		}

		MPI_Barrier(active_comm);
	}

	/* 5. Statistics: Total remaining water and maximum amount of water in a cell */
	max_water_scenario = 0.0;
	for (row_pos = my_start_row; row_pos < my_end_row; row_pos++)
	{
		for (col_pos = 0; col_pos < columns; col_pos++)
		{
			float water = FLOATING(accessMat(water_level, row_pos, col_pos));
			if (water > max_water_scenario)
				max_water_scenario = water;
			total_water += accessMat(water_level, row_pos, col_pos);
		}
	}

	long global_total_water, global_total_water_loss, global_total_rain;
	float global_max_water_scenario;

	MPI_Reduce(&max_water_scenario, &global_max_water_scenario, 1, MPI_FLOAT, MPI_MAX, 0, active_comm);
	MPI_Reduce(&total_water, &global_total_water, 1, MPI_LONG, MPI_SUM, 0, active_comm);
	MPI_Reduce(&total_rain, &global_total_rain, 1, MPI_LONG, MPI_SUM, 0, active_comm);
	MPI_Reduce(&total_water_loss, &global_total_water_loss, 1, MPI_LONG, MPI_SUM, 0, active_comm);

	/* 6. Free resources */
	free(ground);
	free(water_level);
	free(spillage_flag);
	free(spillage_level);
	free(spillage_from_neigh);
	free(clouds);

	// free(up_g);
	// free(up_w);
	// free(down_g);
	// free(down_w);

	free(topHaloSend);
	free(bottomHaloSend);
	free(topHaloRecv);
	free(bottomHaloRecv);

	/*
	 *
	 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
	 *
	 */

	/* 7. Stop global time */
	MPI_Barrier(active_comm);
	ttotal = cp_Wtime() - ttotal;

#ifndef ANIMATION
	/* 8. Output for leaderboard */
	if (mpi_rank == 0)
	{
		/* 8.1. Total computation time */
		printf("\nTime: %lf\n", ttotal);

		/* 8.2. Results: Statistics */
		printf("Result: %d, %d, %10.6lf, %10.6lf, %10.6lf, %10.6lf, %10.6f\n\n",
			   minute,
			   max_spillage_minute,
			   max_spillage_scenario,
			   global_max_water_scenario,
			   FLOATING(global_total_rain),
			   FLOATING(global_total_water),
			   FLOATING(global_total_water_loss));
		printf("Check precision loss: %10.6f\n\n", FLOATING(global_total_rain - global_total_water - global_total_water_loss));
	}
#else
	// This code line does not compute anything usefull
	// It is included to avoid a compilation warning in animation mode
	max_spillage_minute += 0;
#endif

	/* 9. End */
	MPI_Finalize();
	return 0;
}

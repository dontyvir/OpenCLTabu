/*
 * Main.cpp
 *
 *  Created on: 14-08-2012
 *      Author: donty
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <SDKFile.hpp>
#include <SDKCommon.hpp>
#include <SDKApplication.hpp>
#include <unistd.h>
#include "InstanceParser.h"
#include "Solution.h"
#include "SolutionBuilder.h"
#include "Solver.h"
#include "Matrix.h"

int main(int argc, char* argv[]) {

	int iterations;
	int diversification_param;
	int machines;
	int parts;
	int cells;
	int max_machines_cell;
	int tabu_turns;
	std::string filename = "";
	opterr = 0;
	int c;

	if(argc < 13){
		std::cout << "argumentos : -i <numero iteraciones> -d <param diversificacion> -c <celdas> -m <máquinas max por celda> -t <turnos tabu> -f <archivo entrada>\n";
		return EXIT_SUCCESS;
	}

	while ((c = getopt(argc, argv, "i:d:m:p:c:M:t:f:")) != -1){
		switch (c) {
		case 'i':

			iterations = atoi(optarg);
			break;
		case 'd':

			diversification_param = atoi(optarg);
			break;
		case 'm':

			max_machines_cell = atoi(optarg);
			break;
		case 'c':

			cells = atoi(optarg);
			break;
		case 't':

			tabu_turns = atoi(optarg);
			break;
		case 'f':

			filename.assign(optarg, strlen(optarg));
			break;
		case '?':
			if (optopt == 'c')
				fprintf(stderr, "Option -%c requires an argument.\n", optopt);
			else if (isprint(optopt))
				fprintf(stderr, "Unknown option `-%c'.\n", optopt);
			else
				fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
			return 1;
		default:
			std::cout << "argumentos : -i <numero iteraciones> -d <param diversificacion> -c <celdas> -m <máquinas max por celda> -t <turnos tabu>\n";
			abort();
			break;
		}
	}

	tabu::InstanceParser *parser = new tabu::InstanceParser();
	tabu::Matrix *mat = parser->parse_input(filename.c_str());
	machines = parser->machines;
	parts = parser->parts;

	std::cout << " i: " << iterations
			  << " d: " << diversification_param
			  << " m: " << max_machines_cell
			  << " c: " << cells
			  << " t: " << tabu_turns
			  << " f: " << filename
			  << " machines : " << machines
			  << " parts : " << parts << std::endl;


	tabu::Solver *solver = new tabu::Solver(iterations, diversification_param,
			machines, parts, cells, max_machines_cell, mat,
			tabu_turns);
	tabu::Solution *sol = solver->solve();

// ----------------------------------------------------------------------------------

	std::cout << "\n----- global results ------" << std::endl;

	for (int i = 0; i < machines; i++)
		std::cout << sol->cell_vector[i] << " ";

	std::cout << "  cost: " << solver->global_best_cost << std::endl
			<< std::endl;

	int *rows = new int[machines];
	int j = 0;
	for (int k = 0; k < cells; k++) {
		for (int i = 0; i < machines; i++) {
			if (sol->cell_vector[i] == (signed int) k) {
				rows[j] = i;
				j++;
			}
		}
	}

	std::vector<int> columns(parts);
	for (int j = 0; j < parts; j++) {
		columns[j] = j;
	}

	std::vector<int> selected;

	int col = 0;
	for (int i = 0; i < machines; i++) {
		for (int j = 0; j < parts; j++) {

			bool used = false;
			for (unsigned int k = 0; k < selected.size(); k++) {
				if (selected[k] == j)
					used = true;
			}
			if (used == true)
				continue;

			if (col == parts)
				break;

			if (solver->incidence_matrix->getMatrix()[rows[i]][j] == 1) {
				columns[col] = j;
				col++;
				selected.push_back(j);
			}
		}
	}

	std::cout << "\nincidence matrix " << std::endl << std::endl;

	std::cout << "  ";
	for (int j = 0; j < parts; j++) {
		std::cout << j << " ";
	}
	std::cout << std::endl;

	for (int i = 0; i < machines; i++) {

		std::cout << i << " ";

		for (int j = 0; j < parts; j++) {
			int item = solver->incidence_matrix->getMatrix()[i][j];
			if (item == 1)
				std::cout << item << " ";
			else
				std::cout << ". ";
		}
		std::cout << std::endl;
	}

	std::cout << "\nsolution matrix " << std::endl << std::endl;

	std::cout << "      ";
	for (int j = 0; j < parts; j++) {
		std::cout << columns[j] << " ";
	}
	std::cout << std::endl;

	for (int i = 0; i < machines; i++) {

		std::cout << rows[i] << "(k" << sol->cell_vector[rows[i]] << ") ";

		for (int j = 0; j < parts; j++) {
			int item =
					solver->incidence_matrix->getMatrix()[rows[i]][columns[j]];
			if (item == 1)
				std::cout << item << " ";
			else
				std::cout << ". ";
		}
		std::cout << std::endl;
	}

	return SDK_SUCCESS;
}


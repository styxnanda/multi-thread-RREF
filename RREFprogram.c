#include <stdio.h>
#include <omp.h>

void showMatrix(int row, int col, double matrix[][col]);
void swapRows(int threads, int row, int col, double matrix[row][col], int rowA, int rowB);
void divideRow(int threads, int row, int col, double matrix[row][col], int rowNum);
void multiplyRow(int threads, int row, int col, double matrix[row][col], int eliminatedRow, int baseRow);
void reduceMatrix(int threads, int row, int col, double matrix[row][col]);
void setNegativeZeroToZero(int row, int col, double matrix[row][col]);

int main(){
	int kolom, baris, threadUse, i, j;
	
	//input size matrix
	printf("Masukkan jumlah baris: ");
	scanf("%d", &baris);
	printf("Masukkan jumlah kolom: ");
	scanf("%d", &kolom);
	double matrix[baris][kolom];
	
	//input jumlah threads yg digunakan
	printf("Masukkan jumlah thread yang ingin digunakan (Maksimal %d): ", omp_get_max_threads());
	scanf("%d", &threadUse);
	if(threadUse > omp_get_max_threads()){
		printf("Menjalankan dengan thread lebih dari logical processor...\n");
	}
	
	//input elemen-elemen pada matriks
	printf("Masukkan elemen-elemen matrix (pastikan sesuai ukuran!):\n");
	for(i = 0; i < baris; i++){
		for(j = 0; j < kolom; j++){
			scanf("%lf", &matrix[i][j]);
		}
	}
	
	//print matrix awal
	printf("Matrix awal:\n");
	showMatrix(baris, kolom, matrix);
	
	//solve BEBT (RREF) beserta steps
	reduceMatrix(threadUse, baris, kolom, matrix);
	setNegativeZeroToZero(baris, kolom, matrix);
	
	//print matrix hasil
	printf("BEBT (Matrix Hasil):\n");
	showMatrix(baris, kolom, matrix);
	return 0;
}

void reduceMatrix(int threads, int row, int col, double matrix[row][col]){
	int leading, leadingRow = 0, leadingCol = 0, i, j = 0;
	
	//looping tahap 1 s.d. tahap 5 (Eliminasi Gauss)
	while(leadingRow < row && leadingCol < col){
		//swap jika baris awal memiliki elemen nol, sedangkan baris bawahnya memiliki elemen tidak nol
		//(pada kolom yg sama)
		if(matrix[leadingRow][leadingCol] == 0 && leadingRow < row - 1){
			for(i = leadingRow+1; i < row - leadingRow; i++){
				if(matrix[leadingRow+i][leadingCol] != 0){
					swapRows(threads, row, col, matrix, leadingRow, leadingRow+i);
					showMatrix(row, col, matrix);
					break;
				}
			}
		}
		//membagi baris awal dengan pivotnya
		divideRow(threads, row, col, matrix, leadingRow);
		showMatrix(row, col, matrix);
		
		//cari leading one baris awal
		while(matrix[leadingRow][j] == 0){
			j++;
		}
		
		//eliminasi angka di bawah leading one baris awal
		if(leadingRow < row - 1){
			while(matrix[leadingRow][leadingCol] == 0){
				leadingCol += 1;
			}
			for(i = leadingRow+1; i < row; i++){
				if(matrix[i][leadingCol] != 0){
					multiplyRow(threads, row, col, matrix, i, leadingRow);
					showMatrix(row, col, matrix);
				}
			}	
		}
		//increment baris yang perlu dianalisis (tahap 5)
		leadingRow += 1;
	}

	//looping untuk Eliminasi Jordan (tahap 6)
	//esensi eliminasi Jordan adalah melakukan Eliminasi Gauss, tetapi dari belakang
	//Dan tidak perlu khawatir akan pembagian dan swap karena sudah dilakukan pada tahap Gauss
	//Tahap Jordan hanya OBE saja (perkalian baris)
	while(leadingRow > 0 && leadingCol > 0){
		//Decrement leadingRow supaya bisa gerak ke submatriks atas setelah solve
		leadingRow -= 1;
		//Mencari column leading one terakhir
		leadingCol = 0;
		while(matrix[leadingRow][leadingCol] == 0){
			leadingCol += 1;
		}
		//kebalikan operasi OBE di algoritma Gauss sebelumnya
		if(leadingRow > 0){
			for(i = leadingRow-1; i >= 0; i--){
				if(matrix[i][leadingCol] != 0){
					multiplyRow(threads, row, col, matrix, i, leadingRow);
					showMatrix(row, col, matrix);
				}
			}	
		}
	}
}

void showMatrix(int row, int col, double matrix[row][col]){
	int i, j;
	for(i = 0; i < row; i++){
		for(j = 0; j < col; j++){
			printf("%.2lf\t", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void swapRows(int threads, int row, int col, double matrix[row][col], int rowA, int rowB){
	int i, j, id;
	double temp;
	omp_set_num_threads(threads);
	
	#pragma omp parallel for
	// swap sesuai row yg ingin diubah
	for(j = 0; j < col; j++){
		id = omp_get_thread_num();
		temp = matrix[rowA][j];
		matrix[rowA][j] = matrix[rowB][j];
		matrix[rowB][j] = temp;
		printf("Swapping M[%d][%d] with M[%d][%d] using thread #%d\n", rowA, j, rowB, j, id);
	}
}

void divideRow(int threads, int row, int col, double matrix[row][col], int rowNum){
	int i = 0, j = 0, id;
	double divider;
	omp_set_num_threads(threads);
	
	//mencari pivot sebagai pembagi baris supaya bisa ada leading one
	while(matrix[rowNum][j] == 0){
		j++;
	}
	//pembagi baris adalah angka pivot di baris tersebut
	divider = matrix[rowNum][j];
	//jika sudah 1 maka skip saja
	if(matrix[rowNum][j] == 1){
		return;
	}
	
	#pragma omp parallel
	{
		id = omp_get_thread_num();
		#pragma omp for
		for(j = 0; j < col; j++){
			//pembagian oleh pivot shg diperoleh baris dengan leading one
			matrix[rowNum][j] /= divider;
			printf("Dividing M[%d][%d] with %.2lf using thread #%d\n", rowNum, j, divider, id);
		}
	}
}

void multiplyRow(int threads, int row, int col, double matrix[row][col], int eliminatedRow, int baseRow){
	int i = eliminatedRow, j = 0, id;
	double ratio, leading;
	omp_set_num_threads(threads);
	
	//mencari pivot dari baris base
	//baseRow adalah baris yang memiliki leading one (patokan)
	//eliminatedRow adalah baris yang ingin di-eliminasi (angka di bawah leading one baris atas)
	while(matrix[baseRow][j] == 0){
		j++;
	}
	
	//initialisasi leading one baris atas
	leading = matrix[baseRow][j];
	//mencari rasio untuk operasi elementer baris tereliminasi
	ratio = matrix[i][j] / leading;
	
	#pragma omp parallel
	{
		id = omp_get_thread_num();
		#pragma omp for
		//OBEnya menggunakan perkalian rasio
		for(j = 0; j < col; j++){
			matrix[i][j] -= matrix[baseRow][j] * ratio;
			printf("Doing M[%d][%d] -= M[%d][%d] * %.2lf using thread #%d\n", i, j, baseRow, j, ratio, id);
		}
	}
}

//Saat melakukan operasi pada double/float, terdapat dua nilai nol, yaitu -0 dan +0
//Fungsi ini memperbaiki -0 menjadi 0 agar tidak membingungkan user
void setNegativeZeroToZero(int row, int col, double matrix[row][col]){
	int i, j;
	for(i = 0; i < row; i++){
		for(j = 0; j < col; j++){
			//set semua zero (+ atau -) jadi +0
			if(matrix[i][j] == 0) matrix[i][j] = 0;
		}
	}
}

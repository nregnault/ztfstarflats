#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
typedef struct Anode {
   int    n;
   int    nsuper;
   int    nsub;
   int    nnzl;
   int    *xadj;
   int    *adj;
   double *anz;
   double *adiag;
   int    *xsuper;
   int    *snodes;
   int    *xlindx;
   int    *lindx;
   int    *xlnz;
   int *newxlnz;
   double *lnz;
   double *diag;
   double *tmat;
   int    *perm;
   int    *invp;
   int    *colcnt;
   int    *iwork;
   int    tmpsiz;
   int    *split;
   double *newrhs;
}  Anode_type;

extern int ldlt_preprocess__(int *, int *, int *, int *, int *, int *, int *);
extern int ldlt_fact__(int *, int *, int *, double *);
extern int ldlt_selinv__(int *, double *, int*);
extern int ldlt_free__(int *);
extern Anode_type mat[];

void selinv(int * colptr, int * rowind, double * nzvals, double * diag, int nnodes){
  int token = 0, dumpL=0;
  int Lnnz;
  int order=-1;
  int * perm = NULL;
  ldlt_preprocess__(&token, &nnodes, colptr, rowind, &Lnnz, &order, perm);   
  ldlt_fact__(&token, colptr, rowind, nzvals);
  ldlt_selinv__(&token, diag, &dumpL);
  ldlt_free__(&token); 
}

void selinvp(int * colptr, int * rowind, double * nzvals, double * diag, int nnodes, int * perm){
  int token = 0, dumpL=0;
  int Lnnz;
  int order = 0;
  ldlt_preprocess__(&token, &nnodes, colptr, rowind, &Lnnz, &order, perm);   
  ldlt_fact__(&token, colptr, rowind, nzvals);
  ldlt_selinv__(&token, diag, &dumpL);
  ldlt_free__(&token); 
}

int preproc(int * colptr, int * rowind, double * nzvals, int nnodes, int * perm){
  int token = 0;
  int Lnnz, nnzlplus;
  int order = 0;
  ldlt_preprocess__(&token, &nnodes, colptr, rowind, &Lnnz, &order, perm);   
  nnzlplus = ldlt_fact__(&token, colptr, rowind, nzvals);
  return nnzlplus;
}

void selinvpreprocessed(double * diag, int * Acolptr, int * Arowind, double * selAinv, int nnzlplus, int * perm, int * lcolptr, int * lrowind, double * lnz){
  int token =0, dumpL=0, i, j, k, l, super;
  int * cols, * xsuper, *xlindx, *lindx;
  int neqns = mat[token].n;
  // return the L factor
  memcpy(lnz, mat[token].lnz, sizeof(double) * mat[token].nnzl);
  cols = mat[token].xlnz;
  xsuper = mat[token].xsuper;
  lindx =  mat[token].lindx;
  xlindx =  mat[token].xlindx;
  for (j = 0; j < neqns + 1; j++)
    lcolptr[j] = cols[j] - 1;
  k = 0;
  for (super = 0; super < mat[token].nsuper; super++){
    for (j = xsuper[super]; j < xsuper[super + 1]; j++){
      for (l = xlindx[super]; l < xlindx[super + 1]; l++){
        i = lindx[l - 1];
        if (i >= j)
          lrowind[k++] = i - 1;
      }
    }
  }
  assert (k == mat[token].nnzl);
    
  ldlt_selinv__(&token, diag, &dumpL);
  memcpy(selAinv, mat[token].lnz, sizeof(double) * nnzlplus);
  memcpy(perm, mat[token].invp, sizeof(int) * neqns);
  cols = mat[token].newxlnz;
  
  for (j = 0; j < neqns + 1; j++)
    Acolptr[j] = cols[j] - 1;
  k = 0;
  for (super = 0; super < mat[token].nsuper; super++){
    for (j = xsuper[super]; j < xsuper[super + 1]; j++){
      for (l = xlindx[super]; l < xlindx[super + 1]; l++){
        i = lindx[l - 1];
        //if (i >= j)
        Arowind[k++] = i - 1;
      }
    }
  }
  assert (k == nnzlplus);
  ldlt_free__(&token); 
}

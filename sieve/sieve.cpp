#include<math.h>
#include<string.h>
#include<omp.h>
#include<iostream>

using namespace std;

int N = 10000;
int number_threads = 2;

double t = 0.0;

inline long Strike( bool composite[], long i,long stride, long limit ) {
   for(; i <= limit; i += stride)
      composite[i] = true;
   return i;
}

long longCacheUnfriendlySieve( long n ) {
   long count = 0;
   long m = (long)sqrt((double)n);
   bool *composite = new bool[n+1];
   memset( composite, 0, n );
   t = omp_get_wtime();

   for(long i = 2; i <= m; ++i)
      if(!composite[i]) {
         ++count;
         // Strike walks array of size n here.
         Strike( composite, 2*i, i, n );
      }

   for( long i=m+1; i<=n; ++i )
      if( !composite[i] ){
         ++count;
      }

   t = omp_get_wtime() - t;
   delete[] composite;
   return count;
}

long longCacheFriendlySieve( long n ) {
   long count = 0;
   long m = (long)sqrt((double)n); //m=root n
   bool* composite = new bool[n+1];
   memset( composite, 0, n );
   long* factor = new long[m];  //factor holds all primes encountered in first segment(all p)
   long* striker = new long[m]; //holds the first multiple of all primes encountered in first segment (all 2p)
   long n_factor = 0;
   t= omp_get_wtime();
   for( long i=2; i<=m; ++i ) //find all primes in first segment
      if( !composite[i] )
      {
         ++count;
         striker[n_factor] = Strike( composite, 2*i, i, m ); //limit is sqrt n
         factor[n_factor++] = i;
      }
   // Chops sieve into windows of size Å sqrt(n)
   for( long window=m+1; window<=n; window+=m ) {
      long limit = min(window+m-1,n); //either window is of size m or size n
      omp_set_num_threads(number_threads);
      #pragma omp parallel for shared(striker, factor, composite, limit, n_factor) reduction(+:count)
      for( long k=0; k<n_factor; ++k )
         // Strike walks window of size sqrt(n) here.
         striker[k] = Strike( composite, striker[k], factor[k],limit ); //get first multiple of all primes encountered in this segment to use for next segment
      for( long i=window; i<=limit; ++i )
         if( !composite[i] ) //count all primes within one segment
            ++count;
   }
   t = omp_get_wtime() - t;
   delete[] striker;
   delete[] factor;
   delete[] composite;
   return count;
}

int main() {
   cout << "Enter N: ";
   cin >> N;
   cout << "Enter number of threads: ";
   cin >> number_threads;
   //long count = longCacheUnfriendlySieve(N);
   long count = longCacheFriendlySieve(N);
   cout << count << endl;
   cout << "Time : " << t << endl;
}

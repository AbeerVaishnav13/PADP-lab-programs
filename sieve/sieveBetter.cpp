#include <omp.h>
#include<math.h>
#include<string.h>
#include <iostream>

using namespace std;

// process only odd numbers of a specified block
int eratosthenesOddSingleBlock(const int from, const int to)
{
  const int memorySize = (to - from + 1) / 2;
  // initialize
  char* isPrime = new char[memorySize];
  for (int i = 0; i < memorySize; i++)
    isPrime[i] = 1;
  for (int i = 3; i*i <= to; i+=2)
  {
    // >>> UPDATE October 6, 2011
    // skip multiples of three: 9, 15, 21, 27, ...
    if (i >= 3*3 && i % 3 == 0)
      continue;
    // skip multiples of five
    if (i >= 5*5 && i % 5 == 0)
      continue;
    // skip multiples of seven
    if (i >= 7*7 && i % 7 == 0)
      continue;
    // skip multiples of eleven
    if (i >= 11*11 && i % 11 == 0)
      continue;
    // skip multiples of thirteen
    if (i >= 13*13 && i % 13 == 0)
      continue;
    // <<< UPDATE October 6, 2011
    // skip numbers before current slice
    int minJ = ((from+i-1)/i)*i;
    if (minJ < i*i)
      minJ = i*i;
    // start value must be odd
    if ((minJ & 1) == 0)
      minJ += i;
    // find all odd non-primes
    for (int j = minJ; j <= to; j += 2*i)
    {
      int index = j - from;
      isPrime[index/2] = 0;
    }
  }
  // count primes in this block
  int found = 0;
  for (int i = 0; i < memorySize; i++)
    found += isPrime[i];
  // 2 is not odd => include on demand
  if (from <= 2)
    found++;
  delete[] isPrime;
  return found;
}

// process slice-by-slice, odd numbers only
int eratosthenesBlockwise(int lastNumber, int sliceSize, bool useOpenMP)
{
  // enable/disable OpenMP
  omp_set_num_threads(useOpenMP ? omp_get_num_procs() : 1);
  int found = 0;
  // each slices covers ["from" ... "to"], incl. "from" and "to"
#pragma omp parallel for reduction(+:found)
  for (int from = 2; from <= lastNumber; from += sliceSize)
  {
    int to = from + sliceSize;
    if (to > lastNumber)
      to = lastNumber;
    found += eratosthenesOddSingleBlock(from, to);
  }
  return found;
}

int main() {
   int count = eratosthenesBlockwise(250000000, 1, true);
   cout << count << endl;
}

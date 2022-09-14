/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 *
 *        About:
 *          This file contains k-mer speed tests for several Hash Table Types
 *          built using POGGERS. For more verbose testing please see the 
 *          benchmarks folder.
 *
 * ============================================================================
 */




//#include "include/templated_quad_table.cuh"
#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/probing_schemes/linear_probing.cuh>
#include <poggers/probing_schemes/double_hashing.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut.cuh>
#include <poggers/insert_schemes/single_slot_insert.cuh>
#include <poggers/insert_schemes/bucket_insert.cuh>
#include <poggers/insert_schemes/power_of_n.cuh>
#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/shortened_key_val_pair.cuh>
#include <poggers/sizing/default_sizing.cuh>
#include <poggers/sizing/variadic_sizing.cuh>
#include <poggers/tables/base_table.cuh>

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <openssl/rand.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

// #include <warpcore/bloom_filter.cuh>


// using tiny_static_table_4 = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
// using tcqf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tiny_static_table_4>;


// using warpcore_bloom = warpcore::BloomFilter<uint64_t>;


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


uint64_t num_slots_per_p2(uint64_t nitems){

   //uint64_t nitems = .9*(1ULL << nbits);

   //for p=1/100, this is the correct value

   uint64_t nslots = 959*nitems/100;
   printf("using %llu slots\n", nslots);
   return nslots; 

}

template <typename T>
__host__ T * generate_data(uint64_t nitems){


   //malloc space

   T * vals = (T *) malloc(nitems * sizeof(T));


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %llu/%llu\n", to_fill, nitems);

   }

   return vals;
}

template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;




   if (!filter->insert(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){
    //  atomicAdd((unsigned long long int *) misses, 1ULL);
   } 
      //else{

   //    Val test_val = 0;
   //    assert(filter->query(tile, keys[tid], test_val));
   // }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}


template <typename Filter, typename Key, typename Val>
__global__ void speed_remove_kernel(Filter * filter, Key * keys, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;




   if (!filter->remove(tile, keys[tid]) && tile.thread_rank() == 0){
    //  atomicAdd((unsigned long long int *) misses, 1ULL);
   } 
      //else{

   //    Val test_val = 0;
   //    assert(filter->query(tile, keys[tid], test_val));
   // }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

__global__ void count_bf_misses(bool * vals, uint64_t nitems, uint64_t * misses){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid >= nitems) return;


   if (!vals[tid]){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   }
}

template <typename Filter, typename Key, typename Val>
__global__ void speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val val = 0;
   val += 0;

   if (!filter->query(tile,keys[tid], val) && tile.thread_rank() == 0){
   //    atomicAdd((unsigned long long int *) query_misses, 1ULL);
   // } else {

   //    if (val != vals[tid] && tile.thread_rank() == 0){
   //       atomicAdd((unsigned long long int *) query_failures, 1ULL);
   //    }

   }
   //assert(filter->query(tile, keys[tid], val));


}




__global__ void hash_keys(uint64_t * keys, uint64_t * hashes, uint64_t nitems, uint64_t seed){


   uint64_t tid = threadIdx.x + blockIdx.x*blockDim.x;

   if (tid >= nitems ) return;


   poggers::hashers::murmurHasher<uint64_t, 1> my_hasher;

   my_hasher.init(seed);

   hashes[tid] = my_hasher.hash(keys[tid]);

   return;



}



__host__ void test_copy_speed(const std::string& filename, int num_bits, int num_batches, bool first_file){


   using Key = uint64_t;
   using Val = bool;

   //using Filter = tcqf;

   std::cout << "Starting " << filename << " " << num_bits << std::endl;


   uint64_t nitems = (1ULL << num_bits)*.9;

   printf("Starting filter with %llu items to inserts\n", nitems);

   Key * host_keys = generate_data<Key>(nitems);
   //Val * host_vals = generate_data<Val>(nitems);


   Key * fp_keys = generate_data<Key>(nitems);

   Key * dev_keys;

   Key * first_bucket_keys;

   Key * second_bucket_keys;

   Val * dev_vals;




   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*5);
   cudaDeviceSynchronize();

   printf("Data generated\n");

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;
   misses[3] = 0;
   misses[4] = 0;

   //static seed for testing
   //warpcore_bloom bloom_filter(num_slots_per_p2(nitems), 7);

   


   cudaDeviceSynchronize();

   //init timing materials
   std::chrono::duration<double>  * insert_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
   std::chrono::duration<double>  * query_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
   std::chrono::duration<double>  * fp_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));

   uint64_t * batch_amount = (uint64_t *) malloc(num_batches*sizeof(uint64_t));

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);


   for (uint64_t i = 0; i < num_batches; i++){

      uint64_t start_of_batch = i*nitems/num_batches;
      uint64_t items_in_this_batch = (i+1)*nitems/num_batches;

      if (items_in_this_batch > nitems) items_in_this_batch = nitems;

      items_in_this_batch = items_in_this_batch - start_of_batch;


      batch_amount[i] = items_in_this_batch;


      cudaMalloc((void **)& dev_keys, items_in_this_batch*sizeof(Key));
      //cudaMalloc((void **)& dev_vals, items_in_this_batch*sizeof(Val));

      cudaMalloc((void **)&first_bucket_keys, items_in_this_batch*sizeof(Key));

      cudaMalloc((void **)&second_bucket_keys, items_in_this_batch*sizeof(Key));


      cudaMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), cudaMemcpyHostToDevice);
      //cudaMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), cudaMemcpyHostToDevice);



      //ensure GPU is caught up for next task
      cudaDeviceSynchronize();

      auto insert_start = std::chrono::high_resolution_clock::now();

      //add function for configure parameters - should be called by ht and return dim3
      hash_keys<<<(items_in_this_batch-1)/1024+1, 1024>>>(dev_keys, first_bucket_keys, items_in_this_batch, 24);

      hash_keys<<<(items_in_this_batch-1)/1024+1, 1024>>>(dev_keys, second_bucket_keys, items_in_this_batch, 42);

      thrust::sort(thrust::device, first_bucket_keys, first_bucket_keys+items_in_this_batch);
      thrust::sort(thrust::device, second_bucket_keys, second_bucket_keys+items_in_this_batch);


      cudaDeviceSynchronize();
      auto insert_end = std::chrono::high_resolution_clock::now();

      insert_diff[i] = insert_end-insert_start;




      cudaMemcpy(dev_keys, host_keys+start_of_batch, items_in_this_batch*sizeof(Key), cudaMemcpyHostToDevice);
      //cudaMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), cudaMemcpyHostToDevice);


      cudaDeviceSynchronize();




      auto query_start = std::chrono::high_resolution_clock::now();

   
      hash_keys<<<(items_in_this_batch-1)/1024+1, 1024>>>(dev_keys, first_bucket_keys, items_in_this_batch, 24);

      hash_keys<<<(items_in_this_batch-1)/1024+1, 1024>>>(dev_keys, second_bucket_keys, items_in_this_batch, 42);

      thrust::sort(thrust::device, first_bucket_keys, first_bucket_keys+items_in_this_batch);
      thrust::sort(thrust::device, second_bucket_keys, second_bucket_keys+items_in_this_batch);

      cudaDeviceSynchronize();

      auto query_end = std::chrono::high_resolution_clock::now();

      //return;

      query_diff[i] = query_end - query_start;

      //count_bf_misses<<<(items_in_this_batch-1)/1024 +1, 1024>>>(dev_vals, items_in_this_batch, &misses[0]);


      cudaMemcpy(dev_keys, fp_keys+start_of_batch, items_in_this_batch*sizeof(Key), cudaMemcpyHostToDevice);
      //cudaMemcpy(dev_vals, host_vals+start_of_batch, items_in_this_batch*sizeof(Val), cudaMemcpyHostToDevice);


      cudaDeviceSynchronize();

      auto fp_start = std::chrono::high_resolution_clock::now();

      hash_keys<<<(items_in_this_batch-1)/1024+1, 1024>>>(dev_keys, first_bucket_keys, items_in_this_batch, 24);

      hash_keys<<<(items_in_this_batch-1)/1024+1, 1024>>>(dev_keys, second_bucket_keys, items_in_this_batch, 42);

      thrust::sort(thrust::device, first_bucket_keys, first_bucket_keys+items_in_this_batch);
      thrust::sort(thrust::device, second_bucket_keys, second_bucket_keys+items_in_this_batch);


      cudaDeviceSynchronize();
      auto fp_end = std::chrono::high_resolution_clock::now();

      fp_diff[i] = fp_end-fp_start;

      //count_bf_misses<<<(items_in_this_batch-1)/1024 +1, 1024>>>(dev_vals, items_in_this_batch, &misses[1]);

      cudaDeviceSynchronize();

      cudaFree(dev_keys);
      cudaFree(first_bucket_keys);
      cudaFree(second_bucket_keys);


   }

   cudaDeviceSynchronize();


   //Filter::free_on_device(test_filter);

   free(host_keys);
   //free(host_vals);
   free(fp_keys);

   //free pieces

   //time to output


   printf("%llu %llu %f %llu %f %f \n", nitems, misses[0], 1.0*(misses[0])/nitems, misses[1], 1.0*misses[1]/nitems, 1.0 - 1.0*misses[1]/nitems);

   std::chrono::duration<double> summed_insert_diff = std::chrono::nanoseconds::zero();

   for (int i =0; i < num_batches;i++){
      summed_insert_diff += insert_diff[i];
   }

   std::chrono::duration<double> summed_query_diff = std::chrono::nanoseconds::zero();

   for (int i =0; i < num_batches;i++){
      summed_query_diff += query_diff[i];
   }

   std::chrono::duration<double> summed_fp_diff = std::chrono::nanoseconds::zero();

   for (int i =0; i < num_batches;i++){
      summed_fp_diff += fp_diff[i];
   }

   std::string insert_file = filename + "_" + std::to_string(num_bits) + "_insert.txt";
   std::string query_file = filename + "_" + std::to_string(num_bits) + "_lookup.txt";
   std::string fp_file = filename + "_" + std::to_string(num_bits) + "_fp.txt";
   
   //agg files
   std::string agg_insert_file = filename + "_aggregate_inserts.txt";
   std::string agg_lookup_file = filename + "_aggregate_lookup.txt";
   std::string agg_fp_file = filename + "_aggregate_fp.txt";


  // std::cout << insert_file << std::endl;


   FILE *fp_insert = fopen(insert_file.c_str(), "w");
   FILE *fp_lookup = fopen(query_file.c_str(), "w");
   FILE *fp_false_lookup = fopen(fp_file.c_str(), "w");


   FILE *fp_agg_insert;
   FILE *fp_agg_lookup;
   FILE *fp_agg_fp;

   if (first_file){

      fp_agg_insert = fopen(agg_insert_file.c_str(), "w");
      fp_agg_lookup = fopen(agg_lookup_file.c_str(), "w");
      fp_agg_fp = fopen(agg_fp_file.c_str(), "w");

   } else {

      fp_agg_insert = fopen(agg_insert_file.c_str(), "a");
      fp_agg_lookup = fopen(agg_lookup_file.c_str(), "a");
      fp_agg_fp = fopen(agg_fp_file.c_str(), "a");

   }


   if (fp_insert == NULL) {
    printf("Can't open the data file %s\n", insert_file);
    exit(1);
   }

   if (fp_lookup == NULL ) {
      printf("Can't open the data file %s\n", query_file);
      exit(1);
   }

   if (fp_false_lookup == NULL) {
      printf("Can't open the data file %s\n", fp_file);
      exit(1);
   }



   if (fp_agg_insert == NULL || fp_agg_lookup == NULL || fp_agg_fp == NULL) {
      printf("Can't open the aggregate files for %s-%d\n", filename, num_bits);
      exit(1);
   }


   //inserts

   //const uint64_t scaling_factor = 1ULL;
   const uint64_t scaling_factor = 1000000ULL;

   std::cout << "Writing results to file: " <<  insert_file << std::endl;

   fprintf(fp_insert, "x_0 y_0\n");
   for (int i = 0; i < num_batches; i++){
      fprintf(fp_insert, "%d", i*100/num_batches);

      fprintf(fp_insert, " %f\n", batch_amount[i]/(scaling_factor*insert_diff[i].count()));
   }


   //queries
   std::cout << "Writing results to file: " <<  query_file << std::endl;
  

   fprintf(fp_lookup, "x_0 y_0\n");
   for (int i = 0; i < num_batches; i++){
      fprintf(fp_lookup, "%d", i*100/num_batches);

      fprintf(fp_lookup, " %f\n", batch_amount[i]/(scaling_factor*query_diff[i].count()));
   }


   std::cout << "Writing results to file: " <<  fp_file << std::endl;

   fprintf(fp_false_lookup, "x_0 y_0\n");
   for (int i = 0; i < num_batches; i++){
      fprintf(fp_false_lookup, "%d", i*100/num_batches);

      fprintf(fp_false_lookup, " %f\n", batch_amount[i]/(scaling_factor*fp_diff[i].count()));
   }

   if (first_file){
      fprintf(fp_agg_insert, "x_0 y_0\n");
      fprintf(fp_agg_lookup, "x_0 y_0\n");
      fprintf(fp_agg_fp, "x_0 y_0\n");
   }

   // fprintf(fp_agg_insert, "%d %f\n", num_bits, nitems/(1000ULL*summed_insert_diff.count()));
   // fprintf(fp_agg_lookup,  "%d %f\n", num_bits, nitems/(1000ULL*summed_query_diff.count()));
   // fprintf(fp_agg_fp,  "%d %f\n", num_bits, nitems/(1000ULL*summed_fp_diff.count()));

   fprintf(fp_agg_insert, "%d %f\n", num_bits, nitems/(scaling_factor*summed_insert_diff.count()));
   fprintf(fp_agg_lookup,  "%d %f\n", num_bits, nitems/(scaling_factor*summed_query_diff.count()));
   fprintf(fp_agg_fp,  "%d %f\n", num_bits, nitems/(scaling_factor*summed_fp_diff.count()));

   fclose(fp_insert);
   fclose(fp_lookup);
   fclose(fp_false_lookup);
   fclose(fp_agg_insert);
   fclose(fp_agg_lookup);
   fclose(fp_agg_fp);

   return;




}

int main(int argc, char** argv) {

   // poggers::sizing::size_in_num_slots<1> first_size_20(1ULL << 20);
   // printf("2^20\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_20);

   // poggers::sizing::size_in_num_slots<1> first_size_22(1ULL << 22);
   // printf("2^22\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_22);

   // poggers::sizing::size_in_num_slots<1> first_size_24(1ULL << 24);
   // printf("2^24\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_24);

   // poggers::sizing::size_in_num_slots<1> first_size_26(1ULL << 26);
   // printf("2^26\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_26);

   // poggers::sizing::size_in_num_slots<1> first_size_28(1ULL << 28);
   // printf("2^28\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_28);


   // printf("alt table\n");

   // poggers::sizing::size_in_num_slots<1>half_split_20(6000);
   // test_speed<p2_table, key_type, val_type>(&half_split_20);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_22);

   // poggers::sizing::size_in_num_slots<2>half_split_24(1ULL << 23, 1ULL << 23);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_24);

   // poggers::sizing::size_in_num_slots<2>half_split_26(1ULL << 25, 1ULL << 25);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_26);


//   printf("P2 tiny table\n");
   // poggers::sizing::size_in_num_slots<1>half_split_28(1ULL << 28);
   // test_speed<p2_table, key_type, val_type>(&half_split_28);

   // poggers::sizing::variadic_size size(100000,100);
   // tcqf * test_tcqf = tcqf::generate_on_device(&size, 42);


   // cudaDeviceSynchronize();

   // tcqf::free_on_device(test_tcqf);


   // warpcore_bloom my_filter((1ULL << 20), 7);


   // test_bloom_speed("bloom_results/test", 20, 20, true);
   // test_bloom_speed("bloom_results/test", 22, 20, false);
   // test_bloom_speed("bloom_results/test", 24, 20, false);
   // test_bloom_speed("bloom_results/test", 26, 20, false);
   // test_bloom_speed("bloom_results/test", 28, 20, false);
   // test_bloom_speed("bloom_results/test", 30, 20, false);


   // test_tcqf_speed("results/test", 20, 20, true);
   // test_tcqf_speed("results/test", 22, 20, false);
   // test_tcqf_speed("results/test", 24, 20, false);
   // test_tcqf_speed("results/test", 26, 20, false);
   // test_tcqf_speed("results/test", 28, 20, false);
   //test_tcqf_speed("results/test", 30, 20, false);

   test_copy_speed("speed_results/test", 20 , 20, true);
   test_copy_speed("speed_results/test", 22 , 20, false);
   test_copy_speed("speed_results/test", 24 , 20, false);
   test_copy_speed("speed_results/test", 26 , 20, false);
   test_copy_speed("speed_results/test", 28 , 20, false);
   test_copy_speed("speed_results/test", 30 , 20, false);

   test_copy_speed("speed_results/large_test", 20 , 1, true);
   test_copy_speed("speed_results/large_test", 22 , 1, false);
   test_copy_speed("speed_results/large_test", 24 , 1, false);
   test_copy_speed("speed_results/large_test", 26 , 1, false);
   test_copy_speed("speed_results/large_test", 28 , 1, false);
   test_copy_speed("speed_results/large_test", 30 , 1, false);

	return 0;

}

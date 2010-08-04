/*
 * Copyright (c) 2007, Cameron Rich
 * 
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice, 
 *   this list of conditions and the following disclaimer in the documentation 
 *   and/or other materials provided with the distribution.
 * * Neither the name of the axTLS project nor the names of its contributors 
 *   may be used to endorse or promote products derived from this software 
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef O3_CRYPTO_H
#define O3_CRYPTO_H

//#include <sys/timeb.h>
#include <sys/types.h> 
#include <sys/timeb.h>

#ifdef O3_WIN32
 #include <Winsock2.h>
#endif

#undef check
#undef require

namespace o3 {

    namespace Crypto{
    
        /*
         * 
         *   BEGIN Shared crypto utils
         *  
         */

        #ifdef O3_WIN32
            inline void gettimeofday(struct timeval* t, void* /*timezone*/) {       
                struct _timeb timebuffer;

                _ftime(&timebuffer);

                t->tv_sec = (long) timebuffer.time;
                t->tv_usec = 1000 * timebuffer.millitm;
            }
        #endif // O3_WIN32_H

        struct RC4_CTX {
			uint8_t x, y, m[256];
        };

        inline void RC4_setup(RC4_CTX *ctx, const uint8_t *key, int length) {
            int i, j = 0, k = 0, a;
            uint8_t *m;

            ctx->x = 0;
            ctx->y = 0;
            m = ctx->m;

            for (i = 0; i < 256; i++) {
                m[i] = (uint8_t)i;
            }

            for (i = 0; i < 256; i++) {
                a = m[i];
                j = (uint8_t)(j + a + key[k]);
                m[i] = m[j]; 
                m[j] = (uint8_t)a;

                if (++k >= length) {
                    k = 0;
                }
            }
        }

        inline void RC4_crypt(RC4_CTX *ctx, const uint8_t *msg, uint8_t *out, int length) { 
            int i;
            uint8_t *m, x, y, a, b;
            out = (uint8_t *)msg; 

            x = ctx->x;
            y = ctx->y;
            m = ctx->m;

            for (i = 0; i < length; i++) {
                a = m[++x];
                y += a;
                m[x] = b = m[y];
                m[y] = a;
                out[i] ^= m[(uint8_t)(a + b)];
            }

            ctx->x = x;
            ctx->y = y;
        }

        inline void get_random(int num_rand_bytes, uint8_t *rand_data) {   
            static uint64_t rng_num;

            if (num_rand_bytes < 0) {
                rng_num ^= *(uint64_t *) rand_data;

                return;
            }

            RC4_CTX rng_ctx;
            struct timeval tv;
            uint64_t big_num1, big_num2;

            gettimeofday(&tv, NULL);

            big_num1 = (uint64_t)tv.tv_sec*(tv.tv_usec+1); 
            big_num2 = (uint64_t)rand()*big_num1;
            big_num1 ^= rng_num;

            memcpy(rand_data, &big_num1, sizeof(uint64_t));

            if (num_rand_bytes > sizeof(uint64_t)) {
                memcpy(&rand_data[8], &big_num2, sizeof(uint64_t));
            }

            if (num_rand_bytes > 16) {
                memset(&rand_data[16], 0, num_rand_bytes-16); 
            }

            RC4_setup(&rng_ctx, rand_data, 16);
            RC4_crypt(&rng_ctx, rand_data, rand_data, num_rand_bytes);

            memcpy(&rng_num, &rand_data[num_rand_bytes-8], sizeof(uint64_t));    
        }

        inline void get_random_NZ(int num_rand_bytes, uint8_t *rand_data) {
            if (num_rand_bytes < 0) {
                get_random(num_rand_bytes, rand_data);
            } else {
                int i;

                get_random(num_rand_bytes, rand_data);

                for (i = 0; i < num_rand_bytes; i++) {
                    while (rand_data[i] == 0) {
                        rand_data[i] = (uint8_t)(rand());
                    }
                }
            }
        }

        inline void RNG_initialize(const uint8_t *seed_buf, int size) {
            int i;  

            for (i = 0; i < size/(int)sizeof(uint64_t); i++) {
                get_random_NZ(-1, (uint8_t *) &seed_buf[i*sizeof(uint64_t)]);
            }

            srand((long)(int64_t)seed_buf);
        }

        inline void RNG_terminate(void) {
        }
    
    


        /*
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         *   BEGIN Begint library
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         *  
         */

        /* Maintain a number of precomputed variables when doing reduction */
        #define BIGINT_M_OFFSET     0    /**< Normal modulo offset. */
        #define BIGINT_NUM_MODS     1    

        /* Architecture specific functions for big ints */
        #define COMP_RADIX          4294967296ll
        #define COMP_MAX            0xFFFFFFFFFFFFFFFFull
        #define COMP_BIT_SIZE       32  /**< Number of bits in a component. */
        #define COMP_BYTE_SIZE      4   /**< Number of bytes in a component. */
        #define COMP_NUM_NIBBLES    8   /**< Used For diagnostics only. */

        #define PERMANENT           0x7FFF55AA  /**< A magic number for permanents. */

        #define V1   v->comps[v->size-1]                 /**< v1 for division */
        #define V2   v->comps[v->size-2]                 /**< v2 for division */
        #define U(j) tmp_u->comps[tmp_u->size-j-1]       /**< uj for division */
        #define Q(j) quotient->comps[quotient->size-j-1] /**< qj for division */


        typedef uint32_t comp;            /**< A single precision component. */
        typedef uint64_t long_comp;     /**< A double precision component. */
        typedef int64_t slong_comp;     /**< A signed double precision component. */

        /**
         * @struct  _bigint
         * @brief A big integer basic object
         */
        struct _bigint
        {
            struct _bigint* next;       /**< The next bigint in the cache. */
            short size;                 /**< The number of components in this bigint. */
            short max_comps;            /**< The heapsize allocated for this bigint */
            int refs;                   /**< An internal reference count. */
            comp* comps;                /**< A ptr to the actual component data */
        };

        typedef struct _bigint bigint;  /**< An alias for _bigint */

        /**
         * Maintains the state of the cache, and a number of variables used in 
         * reduction.
         */
        typedef struct /**< A big integer "session" context. */
        {
            bigint *active_list;                    /**< Bigints currently used. */
            bigint *free_list;                      /**< Bigints not used. */
            bigint *bi_radix;                       /**< The radix used. */
            bigint *bi_mod[BIGINT_NUM_MODS];        /**< modulus */
            bigint *bi_mu[BIGINT_NUM_MODS];         /**< Storage for mu */
            bigint *bi_normalised_mod[BIGINT_NUM_MODS]; /**< Normalised mod storage. */
            bigint **g;                 /**< Used by sliding-window. */
            int window;                 /**< The size of the sliding window */
            int active_count;           /**< Number of active bigints. */
            int free_count;             /**< Number of free bigints. */

            uint8_t mod_offset;         /**< The mod offset we are using */
        } BI_CTX;

        /**
         * Perform a sanity check on bi.
         */
        inline void check(const bigint *bi) {
            if (bi->refs <= 0) {
                //printf("check: zero or negative refs in bigint\n");
//                o3_assert(false);
            }

            if (bi->next != NULL) {
                //printf("check: attempt to use a bigint from "
                 //       "the free list\n");
//                o3_assert(false);
            }
        }

        /**
         * Allocate and zero more components.  Does not consume bi. 
         */
        inline void more_comps(bigint *bi, int n) {
            if (n > bi->max_comps) {
                bi->max_comps = (short)max(bi->max_comps * 2, n);
                bi->comps = (comp*)realloc(bi->comps, bi->max_comps * COMP_BYTE_SIZE);
            }

            if (n > bi->size) {
                memset(&bi->comps[bi->size], 0, (n-bi->size)*COMP_BYTE_SIZE);
            }

            bi->size = (short)n;
        }

        /**
         * Make a new empty bigint. It may just use an old one if one is available.
         * Otherwise get one off the heap.
         */
        inline bigint *alloc(BI_CTX *ctx, int size) {
            bigint *biR;

            /* Can we recycle an old bigint? */
            if (ctx->free_list != NULL) {
                biR = ctx->free_list;
                ctx->free_list = biR->next;
                ctx->free_count--;

                if (biR->refs != 0) {
                    //printf("alloc: refs was not 0\n");

                    abort();    /* create a stack trace from a core dump */
                }

                more_comps(biR, size);
            }
            else
            {
                /* No free bigints available - create a new one. */
                biR = (bigint *)malloc(sizeof(bigint));
                biR->comps = (comp*)malloc(size * COMP_BYTE_SIZE);
                biR->max_comps = (short)size;  /* give some space to spare */
            }

            biR->size = (short)size;
            biR->refs = 1;
            biR->next = NULL;
            ctx->active_count++;
            return biR;
        }

        /**
         * @brief Simply make a bigint object "unfreeable" if bi_free() is called on it.
         *
         * For this object to be freed, bi_depermanent() must be called.
         * @param bi [in]   The bigint to be made permanent.
         */
        inline void bi_permanent(bigint *bi) {
            check(bi);
            if (bi->refs != 1) {
                //printf("bi_permanent: refs was not 1\n");

                abort();
            }

            bi->refs = PERMANENT;
        }

        /**
         * @brief Take a permanent object and make it eligible for freedom.
         * @param bi [in]   The bigint to be made back to temporary.
         */
        inline void bi_depermanent(bigint *bi) {
            check(bi);
            if (bi->refs != PERMANENT) {
                //printf("bi_depermanent: bigint was not permanent\n");

                abort();
            }

            bi->refs = 1;
        }

        /**
         *@brief Clear the memory cache.
         */
        inline void bi_clear_cache(BI_CTX *ctx) {
            bigint *p, *pn;

            if (ctx->free_list == NULL)
                return;
            
            for (p = ctx->free_list; p != NULL; p = pn) {
                pn = p->next;
                free(p->comps);
                free(p);
            }

            ctx->free_count = 0;
            ctx->free_list = NULL;
        }


        /**
         * @brief Free a bigint object so it can be used again. 
         *
         * The memory itself it not actually freed, just tagged as being available 
         * @param ctx [in]   The bigint session context.
         * @param bi [in]    The bigint to be freed.
         */
        inline void bi_free(BI_CTX *ctx, bigint *bi) {
            check(bi);
            if (bi->refs == PERMANENT) {
                return;
            }

            if (--bi->refs > 0) {
                return;
            }

            bi->next = ctx->free_list;
            ctx->free_list = bi;
            ctx->free_count++;

            if (--ctx->active_count < 0) {
                //printf("bi_free: active_count went negative "
                //        "- double-freed bigint?\n");

                abort();
            }
        }


        /**
         * @brief Start a new bigint context.
         * @return A bigint context.
         */
        inline BI_CTX *bi_initialize(void) {
            /* calloc() sets everything to zero */
            BI_CTX *ctx = (BI_CTX *) calloc(1, sizeof(BI_CTX));
           
            /* the radix */
            ctx->bi_radix = alloc(ctx, 2); 
            ctx->bi_radix->comps[0] = 0;
            ctx->bi_radix->comps[1] = 1;
            
            bi_permanent(ctx->bi_radix);

            return ctx;
        }

        /**
         * @brief Close the bigint context and free any resources.
         *
         * Free up any used memory - a check is done if all objects were not 
         * properly freed.
         * @param ctx [in]   The bigint session context.
         */
        inline void bi_terminate(BI_CTX *ctx) {
            bi_depermanent(ctx->bi_radix); 
            bi_free(ctx, ctx->bi_radix);

            if (ctx->active_count != 0) {
                //printf("bi_terminate: there were %d un-freed bigints\n",
                //               ctx->active_count);

                abort();
            }

            bi_clear_cache(ctx);
            free(ctx);
        }

        /*
         * Delete any leading 0's (and allow for 0).
         */
        inline bigint *trim(bigint *bi) {
            check(bi);

            while (bi->comps[bi->size-1] == 0 && bi->size > 1) {
                bi->size--;
            }

            return bi;
        }

        /**
         * @brief Allow a binary sequence to be imported as a bigint.
         * @param ctx [in]  The bigint session context.
         * @param data [in] The data to be converted.
         * @param size [in] The number of bytes of data.
         * @return A bigint representing this data.
         */
        inline bigint *bi_import(BI_CTX *ctx, const uint8_t *data, int size) {
            bigint *biR = alloc(ctx, (size+COMP_BYTE_SIZE-1)/COMP_BYTE_SIZE);
            int i, j = 0, offset = 0;

            memset(biR->comps, 0, biR->size*COMP_BYTE_SIZE);

            for (i = size-1; i >= 0; i--) {
                biR->comps[offset] += data[i] << (j*8);

                if (++j == COMP_BYTE_SIZE) {
                    j = 0;
                    offset ++;
                }
            }

            return trim(biR);
        }

        /**
         * @brief Take a bigint and convert it into a byte sequence. 
         *
         * This is useful after a decrypt operation.
         * @param ctx [in]  The bigint session context.
         * @param x [in]  The bigint to be converted.
         * @param data [out] The converted data as a byte stream.
         * @param size [in] The maximum size of the byte stream. Unused bytes will be
         * zeroed.
         */
        inline void bi_export(BI_CTX *ctx, bigint *x, uint8_t *data, int size) {
            int i, j, k = size-1;

            check(x);
            memset(data, 0, size);  /* ensure all leading 0's are cleared */

            for (i = 0; i < x->size; i++) {
                for (j = 0; j < COMP_BYTE_SIZE; j++) {
                    comp mask = 0xff << (j*8);
                    int num = (x->comps[i] & mask) >> (j*8);
                    data[k--] = (uint8_t)num;

                    if (k < 0) {
                        break;
                    }
                }
            }

            bi_free(ctx, x);
        }

        /**
         * @brief Convert an (unsigned) integer into a bigint.
         * @param ctx [in]   The bigint session context.
         * @param i [in]     The (unsigned) integer to be converted.
         * 
         */
        inline bigint *int_to_bi(BI_CTX *ctx, comp i) {
            bigint *biR = alloc(ctx, 1);
            biR->comps[0] = i;
            return biR;
        }

        /**
         * @brief Increment the number of references to this object. 
         * It does not do a full copy.
         * @param bi [in]   The bigint to copy.
         * @return A reference to the same bigint.
         */
        inline bigint *bi_copy(bigint *bi) {
            check(bi);
            if (bi->refs != PERMANENT)
                bi->refs++;
            return bi;
        }

        /**
         * @brief Do a full copy of the bigint object.
         * @param ctx [in]   The bigint session context.
         * @param bi  [in]   The bigint object to be copied.
         */
        inline bigint *bi_clone(BI_CTX *ctx, const bigint *bi) {
            bigint *biR = alloc(ctx, bi->size);
            check(bi);
            memcpy(biR->comps, bi->comps, bi->size*COMP_BYTE_SIZE);
            return biR;
        }

        /**
         * @brief Compare two bigints.
         * @param bia [in]  A bigint.
         * @param bib [in]  Another bigint.
         * @return -1 if smaller, 1 if larger and 0 if equal.
         */
        inline int bi_compare(bigint *bia, bigint *bib) {
            int r, i;

            check(bia);
            check(bib);

            if (bia->size > bib->size)
                r = 1;
            else if (bia->size < bib->size)
                r = -1;
            else
            {
                comp *a = bia->comps; 
                comp *b = bib->comps; 

                /* Same number of components.  Compare starting from the high end
                 * and working down. */
                r = 0;
                i = bia->size - 1;

                do 
                {
                    if (a[i] > b[i]) { 
                        r = 1;
                        break; 
                    }
                    else if (a[i] < b[i]) { 
                        r = -1;
                        break; 
                    }
                } while (--i >= 0);
            }

            return r;
        }

        /**
         * Take each component and shift it up (in terms of components) 
         */
        inline static bigint *comp_left_shift(bigint *biR, int num_shifts) {
            int i = biR->size-1;
            comp *x, *y;

            check(biR);

            if (num_shifts <= 0) {
                return biR;
            }

            more_comps(biR, biR->size + num_shifts);

            x = &biR->comps[i+num_shifts];
            y = &biR->comps[i];

            do
            {
                *x-- = *y--;
            } while (i--);

            memset(biR->comps, 0, num_shifts*COMP_BYTE_SIZE); /* zero LS comps */
            return biR;
        }

        /**
         * Take each component and shift down (in terms of components) 
         */
        inline bigint *comp_right_shift(bigint *biR, int num_shifts) {
            int i = biR->size-num_shifts;
            comp *x = biR->comps;
            comp *y = &biR->comps[num_shifts];

            check(biR);

            if (i <= 0)     /* have we completely right shifted? */
            {
                biR->comps[0] = 0;  /* return 0 */
                biR->size = 1;
                return biR;
            }

            do
            {
                *x++ = *y++;
            } while (--i > 0);

            biR->size -= (short)num_shifts;
            return biR;
        }

        /**
         * @brief Perform an addition operation between two bigints.
         * @param ctx [in]  The bigint session context.
         * @param bia [in]  A bigint.
         * @param bib [in]  Another bigint.
         * @return The result of the addition.
         */
        inline bigint *bi_add(BI_CTX *ctx, bigint *bia, bigint *bib) {
            int n;
            comp carry = 0;
            comp *pa, *pb;

            check(bia);
            check(bib);

            n = max(bia->size, bib->size);
            more_comps(bia, n+1);
            more_comps(bib, n);
            pa = bia->comps;
            pb = bib->comps;

            do
            {
                comp  sl, rl, cy1;
                sl = *pa + *pb++;
                rl = sl + carry;
                cy1 = sl < *pa;
                carry = cy1 | (rl < sl);
                *pa++ = rl;
            } while (--n != 0);

            *pa = carry;                  /* do overflow */
            bi_free(ctx, bib);
            return trim(bia);
        }


        /**
         * @brief Perform a subtraction operation between two bigints.
         * @param ctx [in]  The bigint session context.
         * @param bia [in]  A bigint.
         * @param bib [in]  Another bigint.
         * @param is_negative [out] If defined, indicates that the result was negative.
         * is_negative may be null.
         * @return The result of the subtraction. The result is always positive.
         */
        inline bigint *bi_subtract(BI_CTX *ctx, 
                bigint *bia, bigint *bib, int *is_negative) {
            int n = bia->size;
            comp *pa, *pb, carry = 0;

            check(bia);
            check(bib);

            more_comps(bib, n);
            pa = bia->comps;
            pb = bib->comps;

            do 
            {
                comp sl, rl, cy1;
                sl = *pa - *pb++;
                rl = sl - carry;
                cy1 = sl > *pa;
                carry = cy1 | (rl > sl);
                *pa++ = rl;
            } while (--n != 0);

            if (is_negative)    /* indicate a negative result */
            {
                *is_negative = carry;
            }

            bi_free(ctx, trim(bib));    /* put bib back to the way it was */
            return trim(bia);
        }

        /**
         * Perform a multiply between a bigint an an (unsigned) integer
         */
        inline bigint *bi_int_multiply(BI_CTX *ctx, bigint *bia, comp b) {
            int j = 0, n = bia->size;
            bigint *biR = alloc(ctx, n + 1);
            comp carry = 0;
            comp *r = biR->comps;
            comp *a = bia->comps;

            check(bia);

            /* clear things to start with */
            memset(r, 0, ((n+1)*COMP_BYTE_SIZE));

            do
            {
                long_comp tmp = *r + (long_comp)a[j]*b + carry;
                *r++ = (comp)tmp;              /* downsize */
                carry = (comp)(tmp >> COMP_BIT_SIZE);
            } while (++j < n);

            *r = carry;
            bi_free(ctx, bia);
            return trim(biR);
        }

        /** 
         * Perform a standard multiplication between two bigints.
         */
        inline bigint *regular_multiply(BI_CTX *ctx, bigint *bia, bigint *bib) {
            int i, j, i_plus_j;
            int n = bia->size; 
            int t = bib->size;
            bigint *biR = alloc(ctx, n + t);
            comp *sr = biR->comps;
            comp *sa = bia->comps;
            comp *sb = bib->comps;

            check(bia);
            check(bib);

            /* clear things to start with */
            memset(biR->comps, 0, ((n+t)*COMP_BYTE_SIZE));
            i = 0;

            do 
            {
                comp carry = 0;
                comp b = *sb++;
                i_plus_j = i;
                j = 0;

                do
                {
                    long_comp tmp = sr[i_plus_j] + (long_comp)sa[j]*b + carry;
                    sr[i_plus_j++] = (comp)tmp;              /* downsize */
                    carry = (comp)(tmp >> COMP_BIT_SIZE);
                } while (++j < n);

                sr[i_plus_j] = carry;
            } while (++i < t);

            bi_free(ctx, bia);
            bi_free(ctx, bib);
            return trim(biR);
        }

        /**
         * @brief Perform a multiplication operation between two bigints.
         * @param ctx [in]  The bigint session context.
         * @param bia [in]  A bigint.
         * @param bib [in]  Another bigint.
         * @return The result of the multiplication.
         */
        inline bigint *bi_multiply(BI_CTX *ctx, bigint *bia, bigint *bib) {
            check(bia);
            check(bib);

            return regular_multiply(ctx, bia, bib);
        }

        /*
         * Perform an integer divide on a bigint.
         */
        inline bigint *bi_int_divide(BI_CTX* /*ctx*/, bigint *biR, comp denom) {
            int i = biR->size - 1;
            long_comp r = 0;

            check(biR);

            do
            {
                r = (r<<COMP_BIT_SIZE) + biR->comps[i];
                biR->comps[i] = (comp)(r / denom);
                r %= denom;
            } while (--i >= 0);

            return trim(biR);
        }

        /**
         * @brief Does both division and modulo calculations. 
         *
         * Used extensively when doing classical reduction.
         * @param ctx [in]  The bigint session context.
         * @param u [in]    A bigint which is the numerator.
         * @param v [in]    Either the denominator or the modulus depending on the mode.
         * @param is_mod [n] Determines if this is a normal division (0) or a reduction
         * (1).
         * @return  The result of the division/reduction.
         */
        inline bigint *bi_divide(BI_CTX *ctx, bigint *u, bigint *v, int is_mod) {
            int n = v->size, m = u->size-n;
            int j = 0, orig_u_size = u->size;
            uint8_t mod_offset = ctx->mod_offset;
            comp d;
            bigint *quotient, *tmp_u;
            comp q_dash;

            check(u);
            check(v);

            /* if doing reduction and we are < mod, then return mod */
            if (is_mod && bi_compare(v, u) > 0) {
                bi_free(ctx, v);
                return u;
            }

            quotient = alloc(ctx, m+1);
            tmp_u = alloc(ctx, n+1);
            v = trim(v);        /* make sure we have no leading 0's */
            d = (comp)((long_comp)COMP_RADIX/(V1+1));

            /* clear things to start with */
            memset(quotient->comps, 0, ((quotient->size)*COMP_BYTE_SIZE));

            /* normalise */
            if (d > 1) {
                u = bi_int_multiply(ctx, u, d);

                if (is_mod) {
                    v = ctx->bi_normalised_mod[mod_offset];
                }
                else
                {
                    v = bi_int_multiply(ctx, v, d);
                }
            }

            if (orig_u_size == u->size)  /* new digit position u0 */
            {
                more_comps(u, orig_u_size + 1);
            }

            do
            {
                /* get a temporary short version of u */
                memcpy(tmp_u->comps, &u->comps[u->size-n-1-j], (n+1)*COMP_BYTE_SIZE);

                /* calculate q' */
                if (U(0) == V1) {
                    q_dash = COMP_RADIX-1;
                }
                else
                {
                    q_dash = (comp)(((long_comp)U(0)*COMP_RADIX + U(1))/V1);
                }

                if (v->size > 1 && V2) {
                    /* we are implementing the following:
                    if (V2*q_dash > (((U(0)*COMP_RADIX + U(1) - 
                            q_dash*V1)*COMP_RADIX) + U(2))) ... */
                    comp inner = (comp)((long_comp)COMP_RADIX*U(0) + U(1) - 
                                                (long_comp)q_dash*V1);
                    if ((long_comp)V2*q_dash > (long_comp)inner*COMP_RADIX + U(2)) {
                        q_dash--;
                    }
                }

                /* multiply and subtract */
                if (q_dash) {
                    int is_negative;
                    tmp_u = bi_subtract(ctx, tmp_u, 
                            bi_int_multiply(ctx, bi_copy(v), q_dash), &is_negative);
                    more_comps(tmp_u, n+1);

                    Q(j) = q_dash; 

                    /* add back */
                    if (is_negative) {
                        Q(j)--;
                        tmp_u = bi_add(ctx, tmp_u, bi_copy(v));

                        /* lop off the carry */
                        tmp_u->size--;
                        v->size--;
                    }
                }
                else
                {
                    Q(j) = 0; 
                }

                /* copy back to u */
                memcpy(&u->comps[u->size-n-1-j], tmp_u->comps, (n+1)*COMP_BYTE_SIZE);
            } while (++j <= m);

            bi_free(ctx, tmp_u);
            bi_free(ctx, v);

            if (is_mod)     /* get the remainder */
            {
                bi_free(ctx, quotient);
                return bi_int_divide(ctx, trim(u), d);
            }
            else            /* get the quotient */
            {
                bi_free(ctx, u);
                return trim(quotient);
            }
        }

        /*
         * Perform the actual square operion. It takes into account overflow.
         */
        inline bigint *regular_square(BI_CTX *ctx, bigint *bi) {
            int t = bi->size;
            int i = 0, j;
            bigint *biR = alloc(ctx, t*2);
            comp *w = biR->comps;
            comp *x = bi->comps;
            comp carry;

            memset(w, 0, biR->size*COMP_BYTE_SIZE);

            do
            {
                long_comp tmp = w[2*i] + (long_comp)x[i]*x[i];
                comp u = 0;
                w[2*i] = (comp)tmp;
                carry = (comp)(tmp >> COMP_BIT_SIZE);

                for (j = i+1; j < t; j++) {
                    long_comp xx = (long_comp)x[i]*x[j];
                    long_comp xx2 = 2*xx;
                    long_comp blob = (long_comp)w[i+j]+carry;

                    if (u)                  /* previous overflow */
                    {
                        blob += COMP_RADIX;
                    }


                    u = 0;
                    tmp = xx2 + blob;

                    /* check for overflow */
                    if ((COMP_MAX-xx) < xx  || (COMP_MAX-xx2) < blob) {
                        u = 1;
                    }

                    w[i+j] = (comp)tmp;
                    carry = (comp)(tmp >> COMP_BIT_SIZE);
                }

                w[i+t] += carry;

                if (u) {
                    w[i+t+1] = 1;   /* add carry */
                }
            } while (++i < t);

            bi_free(ctx, bi);
            return trim(biR);
        }

        /**
         * @brief Perform a square operation on a bigint.
         * @param ctx [in]  The bigint session context.
         * @param bia [in]  A bigint.
         * @return The result of the multiplication.
         */
        inline bigint *bi_square(BI_CTX *ctx, bigint *bia) {
            check(bia);

            return regular_square(ctx, bia);
        }

        /**
         * @def bi_mod
         * Find the residue of B. bi_set_mod() must be called before hand.
         */
        #define bi_mod(A, B) bi_divide(A, B, ctx->bi_mod[ctx->mod_offset], 1)

        /*
         * Barrett reduction has no need for some parts of the product, so ignore bits
         * of the multiply. This routine gives Barrett its big performance
         * improvements over Classical/Montgomery reduction methods. 
         */
        inline bigint *partial_multiply(BI_CTX *ctx, bigint *bia, bigint *bib, 
                int inner_partial, int outer_partial) {
            int i = 0, j, n = bia->size, t = bib->size;
            bigint *biR;
            comp carry;
            comp *sr, *sa, *sb;

            check(bia);
            check(bib);

            biR = alloc(ctx, n + t);
            sa = bia->comps;
            sb = bib->comps;
            sr = biR->comps;

            if (inner_partial) {
                memset(sr, 0, inner_partial*COMP_BYTE_SIZE); 
            }
            else    /* outer partial */
            {
                if (n < outer_partial || t < outer_partial) /* should we bother? */
                {
                    bi_free(ctx, bia);
                    bi_free(ctx, bib);
                    biR->comps[0] = 0;      /* return 0 */
                    biR->size = 1;
                    return biR;
                }

                memset(&sr[outer_partial], 0, (n+t-outer_partial)*COMP_BYTE_SIZE);
            }

            do 
            {
                comp *a = sa;
                comp b = *sb++;
                long_comp tmp;
                int i_plus_j = i;
                carry = 0;
                j = n;

                if (outer_partial && i_plus_j < outer_partial) {
                    i_plus_j = outer_partial;
                    a = &sa[outer_partial-i];
                    j = n-(outer_partial-i);
                }

                do
                {
                    if (inner_partial && i_plus_j >= inner_partial) {
                        break;
                    }

                    tmp = sr[i_plus_j] + ((long_comp)*a++)*b + carry;
                    sr[i_plus_j++] = (comp)tmp;              /* downsize */
                    carry = (comp)(tmp >> COMP_BIT_SIZE);
                } while (--j != 0);

                sr[i_plus_j] = carry;
            } while (++i < t);

            bi_free(ctx, bia);
            bi_free(ctx, bib);
            return trim(biR);
        }

        /*
         * Stomp on the most significant components to give the illusion of a "mod base
         * radix" operation 
         */
        inline bigint *comp_mod(bigint *bi, int mod) {
            check(bi);

            if (bi->size > mod) {
                bi->size = (short)mod;
            }

            return bi;
        }

        /**
         * @brief Perform a single Barrett reduction.
         * @param ctx [in]  The bigint session context.
         * @param bi [in]  A bigint.
         * @return The result of the Barrett reduction.
         */
        inline bigint *bi_barrett(BI_CTX *ctx, bigint *bi) {
            bigint *q1, *q2, *q3, *r1, *r2, *r;
            uint8_t mod_offset = ctx->mod_offset;
            bigint *bim = ctx->bi_mod[mod_offset];
            int k = bim->size;

            check(bi);
            check(bim);

            /* use Classical method instead  - Barrett cannot help here */
            if (bi->size > k*2) {
                return bi_mod(ctx, bi);
            }

            q1 = comp_right_shift(bi_clone(ctx, bi), k-1);

            /* do outer partial multiply */
            q2 = partial_multiply(ctx, q1, ctx->bi_mu[mod_offset], 0, k-1); 
            q3 = comp_right_shift(q2, k+1);
            r1 = comp_mod(bi, k+1);

            /* do inner partial multiply */
            r2 = comp_mod(partial_multiply(ctx, q3, bim, k+1, 0), k+1);
            r = bi_subtract(ctx, r1, r2, NULL);

            /* if (r >= m) r = r - m; */
            if (bi_compare(r, bim) >= 0) {
                r = bi_subtract(ctx, r, bim, NULL);
            }

            return r;
        }
        #define bi_residue(A, B) bi_barrett(A, B)

        /*
         * Work out the highest '1' bit in an exponent. Used when doing sliding-window
         * exponentiation.
         */
        inline int find_max_exp_index(bigint *biexp) {
            int i = COMP_BIT_SIZE-1;
            comp shift = COMP_RADIX/2;
            comp test = biexp->comps[biexp->size-1];    /* assume no leading zeroes */

            check(biexp);

            do
            {
                if (test & shift) {
                    return i+(biexp->size-1)*COMP_BIT_SIZE;
                }

                shift >>= 1;
            } while (--i != 0);

            return -1;      /* error - must have been a leading 0 */
        }

        /*
         * Is a particular bit is an exponent 1 or 0? Used when doing sliding-window
         * exponentiation.
         */
        inline int exp_bit_is_one(bigint *biexp, int offset) {
            comp test = biexp->comps[offset / COMP_BIT_SIZE];
            int num_shifts = offset % COMP_BIT_SIZE;
            comp shift = 1;
            int i;

            check(biexp);

            for (i = 0; i < num_shifts; i++) {
                shift <<= 1;
            }

            return test & shift;
        }

        /*
         * Work out g1, g3, g5, g7... etc for the sliding-window algorithm 
         */
        inline void precompute_slide_window(BI_CTX *ctx, int window, bigint *g1) {
            int k = 1, i;
            bigint *g2;

            for (i = 0; i < window-1; i++)   /* compute 2^(window-1) */
            {
                k <<= 1;
            }

            ctx->g = (bigint **)malloc(k*sizeof(bigint *));
            ctx->g[0] = bi_clone(ctx, g1);
            bi_permanent(ctx->g[0]);
            g2 = bi_residue(ctx, bi_square(ctx, ctx->g[0]));   /* g^2 */

            for (i = 1; i < k; i++) {
                ctx->g[i] = bi_residue(ctx, bi_multiply(ctx, ctx->g[i-1], bi_copy(g2)));
                bi_permanent(ctx->g[i]);
            }

            bi_free(ctx, g2);
            ctx->window = k;
        }

        /**
         * @brief Pre-calculate some of the expensive steps in reduction. 
         *
         * This function should only be called once (normally when a session starts).
         * When the session is over, bi_free_mod() should be called. bi_mod_power()
         * relies on this function being called.
         * @param ctx [in]  The bigint session context.
         * @param bim [in]  The bigint modulus that will be used.
         * @param mod_offset [in] There are three moduluii that can be stored - the
         * standard modulus, and its two primes p and q. This offset refers to which
         * modulus we are referring to.
         * @see bi_free_mod(), bi_mod_power().
         */
        inline void bi_set_mod(BI_CTX *ctx, bigint *bim, int mod_offset) {
            int k = bim->size;
            comp d = (comp)((long_comp)COMP_RADIX/(bim->comps[k-1]+1));

            ctx->bi_mod[mod_offset] = bim;
            bi_permanent(ctx->bi_mod[mod_offset]);
            ctx->bi_normalised_mod[mod_offset] = bi_int_multiply(ctx, bim, d);
            bi_permanent(ctx->bi_normalised_mod[mod_offset]);


            ctx->bi_mu[mod_offset] = 
                bi_divide(ctx, comp_left_shift(
                    bi_clone(ctx, ctx->bi_radix), k*2-1), ctx->bi_mod[mod_offset], 0);
            bi_permanent(ctx->bi_mu[mod_offset]);
        }

        /**
         * @brief Used when cleaning various bigints at the end of a session.
         * @param ctx [in]  The bigint session context.
         * @param mod_offset [in] The offset to use.
         * @see bi_set_mod().
         */
        inline void bi_free_mod(BI_CTX *ctx, int mod_offset) {
            bi_depermanent(ctx->bi_mod[mod_offset]);
            bi_free(ctx, ctx->bi_mod[mod_offset]);
            bi_depermanent(ctx->bi_mu[mod_offset]); 
            bi_free(ctx, ctx->bi_mu[mod_offset]);
            bi_depermanent(ctx->bi_normalised_mod[mod_offset]); 
            bi_free(ctx, ctx->bi_normalised_mod[mod_offset]);
        }


        /**
         * @brief Perform a modular exponentiation.
         *
         * This function requires bi_set_mod() to have been called previously. This is 
         * one of the optimisations used for performance.
         * @param ctx [in]  The bigint session context.
         * @param bi  [in]  The bigint on which to perform the mod power operation.
         * @param biexp [in] The bigint exponent.
         * @return The result of the mod exponentiation operation
         * @see bi_set_mod().
         */
        inline bigint *bi_mod_power(BI_CTX *ctx, bigint *bi, bigint *biexp) {
            int i = find_max_exp_index(biexp), j, window_size = 1;
            bigint *biR = int_to_bi(ctx, 1);

            check(bi);
            check(biexp);

            for (j = i; j > 32; j /= 5) /* work out an optimum size */
                window_size++;

            /* work out the slide constants */
            precompute_slide_window(ctx, window_size, bi);

            /* if sliding-window is off, then only one bit will be done at a time and
             * will reduce to standard left-to-right exponentiation */
            do
            {
                if (exp_bit_is_one(biexp, i)) {
                    int l = i-window_size+1;
                    int part_exp = 0;

                    if (l < 0)  /* LSB of exponent will always be 1 */
                        l = 0;
                    else
                    {
                        while (exp_bit_is_one(biexp, l) == 0)
                            l++;    /* go back up */
                    }

                    /* build up the section of the exponent */
                    for (j = i; j >= l; j--) {
                        biR = bi_residue(ctx, bi_square(ctx, biR));
                        if (exp_bit_is_one(biexp, j))
                            part_exp++;

                        if (j != l)
                            part_exp <<= 1;
                    }

                    part_exp = (part_exp-1)/2;  /* adjust for array */
                    biR = bi_residue(ctx, bi_multiply(ctx, biR, ctx->g[part_exp]));
                    i = l-1;
                }
                else    /* square it */
                {
                    biR = bi_residue(ctx, bi_square(ctx, biR));
                    i--;
                }
            } while (i >= 0);
             
            /* cleanup */
            for (i = 0; i < ctx->window; i++) {
                bi_depermanent(ctx->g[i]);
                bi_free(ctx, ctx->g[i]);
            }

            free(ctx->g);
            bi_free(ctx, bi);
            bi_free(ctx, biexp);
            return biR;
        }

        /**
         * @brief Perform a modular exponentiation using a temporary modulus.
         *
         * We need this function to check the signatures of certificates. The modulus
         * of this function is temporary as it's just used for authentication.
         * @param ctx [in]  The bigint session context.
         * @param bi  [in]  The bigint to perform the exp/mod.
         * @param bim [in]  The temporary modulus.
         * @param biexp [in] The bigint exponent.
         * @return The result of the mod exponentiation operation
         * @see bi_set_mod().
         */
        inline bigint *bi_mod_power2(BI_CTX *ctx, bigint *bi, bigint *bim, bigint *biexp) {
            bigint *biR, *tmp_biR;

            /* Set up a temporary bigint context and transfer what we need between
             * them. We need to do this since we want to keep the original modulus
             * which is already in this context. This operation is only called when
             * doing peer verification, and so is not expensive :-) */
            BI_CTX *tmp_ctx = bi_initialize();
            bi_set_mod(tmp_ctx, bi_clone(tmp_ctx, bim), BIGINT_M_OFFSET);
            tmp_biR = bi_mod_power(tmp_ctx, 
                        bi_clone(tmp_ctx, bi), 
                        bi_clone(tmp_ctx, biexp));
            biR = bi_clone(ctx, tmp_biR);
            bi_free(tmp_ctx, tmp_biR);
            bi_free_mod(tmp_ctx, BIGINT_M_OFFSET);
            bi_terminate(tmp_ctx);

            bi_free(ctx, bi);
            bi_free(ctx, bim);
            bi_free(ctx, biexp);
            return biR;
        }

        inline void bi_print(const char* /*label*/, bigint* /*x*/) {
          /*  int i, j;

            if (x == NULL) {
                //printf("%s: (null)\n", label);
                return;
            }

            //printf("%s: (size %d)\n", label, x->size);
            for (i = x->size-1; i >= 0; i--) {
                for (j = COMP_NUM_NIBBLES-1; j >= 0; j--) {
                    comp mask = 0x0f << (j*4);
                    comp num = (x->comps[i] & mask) >> (j*4);
                    putc((num <= 9) ? (num + '0') : (num + 'A' - 10), stdout);
                }
            }  

            //printf("\n");*/
        }

        /**
         * @brief The testharness uses this code to import text hex-streams and 
         * convert them into bigints.
         * @param ctx [in]  The bigint session context.
         * @param data [in] A string consisting of hex characters. The characters must
         * be in upper case.
         * @return A bigint representing this data.
         */
        inline bigint *bi_str_import(BI_CTX *ctx, const char *data) {
            int size = (int)strlen(data);
            bigint *biR = alloc(ctx, (size+COMP_NUM_NIBBLES-1)/COMP_NUM_NIBBLES);
            int i, j = 0, offset = 0;
            memset(biR->comps, 0, biR->size*COMP_BYTE_SIZE);

            for (i = size-1; i >= 0; i--) {
                int num = (data[i] <= '9') ? (data[i] - '0') : (data[i] - 'A' + 10);
                biR->comps[offset] += num << (j*4);

                if (++j == COMP_NUM_NIBBLES) {
                    j = 0;
                    offset ++;
                }
            }

            return biR;
        }
 

        /*
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         *   BEGIN SHA1 CODE
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         *  
         */
        
        
         /*
         *  This structure will hold context information for the SHA-1
         *  hashing operation
         */
         
        #define SHA1_SIZE   20
      
        typedef struct 
        {
            uint32_t Intermediate_Hash[SHA1_SIZE/4]; /* Message Digest */
            uint32_t Length_Low;            /* Message length in bits */
            uint32_t Length_High;           /* Message length in bits */
            uint16_t Message_Block_Index;   /* Index into message block array   */
            uint8_t Message_Block[64];      /* 512-bit message blocks */
        } SHA1_CTX;

        inline void SHA1_Init(SHA1_CTX *ctx) {
            ctx->Length_Low             = 0;
            ctx->Length_High            = 0;
            ctx->Message_Block_Index    = 0;
            ctx->Intermediate_Hash[0]   = 0x67452301;
            ctx->Intermediate_Hash[1]   = 0xEFCDAB89;
            ctx->Intermediate_Hash[2]   = 0x98BADCFE;
            ctx->Intermediate_Hash[3]   = 0x10325476;
            ctx->Intermediate_Hash[4]   = 0xC3D2E1F0;
        }

        /*
         *  Define the SHA1 circular left shift macro
         */
        #define SHA1CircularShift(bits,word) \
                        (((word) << (bits)) | ((word) >> (32-(bits))))

        /**
         * Process the next 512 bits of the message stored in the array.
         */
        inline void SHA1ProcessMessageBlock(SHA1_CTX *ctx) {
            const uint32_t K[] =    {       /* Constants defined in SHA-1   */
                                    0x5A827999,
                                    0x6ED9EBA1,
                                    0x8F1BBCDC,
                                    0xCA62C1D6
                                    };
            int        t;                 /* Loop counter                */
            uint32_t      temp;              /* Temporary word value        */
            uint32_t      W[80];             /* Word sequence               */
            uint32_t      A, B, C, D, E;     /* Word buffers                */

            /*
             *  Initialize the first 16 words in the array W
             */
            for  (t = 0; t < 16; t++) {
                W[t] = ctx->Message_Block[t * 4] << 24;
                W[t] |= ctx->Message_Block[t * 4 + 1] << 16;
                W[t] |= ctx->Message_Block[t * 4 + 2] << 8;
                W[t] |= ctx->Message_Block[t * 4 + 3];
            }

            for (t = 16; t < 80; t++) {
               W[t] = SHA1CircularShift(1,W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16]);
            }

            A = ctx->Intermediate_Hash[0];
            B = ctx->Intermediate_Hash[1];
            C = ctx->Intermediate_Hash[2];
            D = ctx->Intermediate_Hash[3];
            E = ctx->Intermediate_Hash[4];

            for (t = 0; t < 20; t++) {
                temp =  SHA1CircularShift(5,A) +
                        ((B & C) | ((~B) & D)) + E + W[t] + K[0];
                E = D;
                D = C;
                C = SHA1CircularShift(30,B);

                B = A;
                A = temp;
            }

            for (t = 20; t < 40; t++) {
                temp = SHA1CircularShift(5,A) + (B ^ C ^ D) + E + W[t] + K[1];
                E = D;
                D = C;
                C = SHA1CircularShift(30,B);
                B = A;
                A = temp;
            }

            for (t = 40; t < 60; t++) {
                temp = SHA1CircularShift(5,A) +
                       ((B & C) | (B & D) | (C & D)) + E + W[t] + K[2];
                E = D;
                D = C;
                C = SHA1CircularShift(30,B);
                B = A;
                A = temp;
            }

            for (t = 60; t < 80; t++) {
                temp = SHA1CircularShift(5,A) + (B ^ C ^ D) + E + W[t] + K[3];
                E = D;
                D = C;
                C = SHA1CircularShift(30,B);
                B = A;
                A = temp;
            }

            ctx->Intermediate_Hash[0] += A;
            ctx->Intermediate_Hash[1] += B;
            ctx->Intermediate_Hash[2] += C;
            ctx->Intermediate_Hash[3] += D;
            ctx->Intermediate_Hash[4] += E;
            ctx->Message_Block_Index = 0;
        }

        /**
         * Accepts an array of octets as the next portion of the message.
         */
        inline void SHA1_Update(SHA1_CTX *ctx, const uint8_t *msg, int len) {
            while (len--) {
                ctx->Message_Block[ctx->Message_Block_Index++] = (*msg & 0xFF);
                ctx->Length_Low += 8;

                if (ctx->Length_Low == 0)
                    ctx->Length_High++;

                if (ctx->Message_Block_Index == 64)
                    SHA1ProcessMessageBlock(ctx);

                msg++;
            }
        }

        /*
         * According to the standard, the message must be padded to an even
         * 512 bits.  The first padding bit must be a '1'.  The last 64
         * bits represent the length of the original message.  All bits in
         * between should be 0.  This function will pad the message
         * according to those rules by filling the Message_Block array
         * accordingly.  It will also call the ProcessMessageBlock function
         * provided appropriately.  When it returns, it can be assumed that
         * the message digest has been computed.
         *
         * @param ctx [in, out] The SHA1 context
         */
        inline void SHA1PadMessage(SHA1_CTX *ctx) {
            /*
             *  Check to see if the current message block is too small to hold
             *  the initial padding bits and length.  If so, we will pad the
             *  block, process it, and then continue padding into a second
             *  block.
             */
            if (ctx->Message_Block_Index > 55) {
                ctx->Message_Block[ctx->Message_Block_Index++] = 0x80;
                while(ctx->Message_Block_Index < 64) {
                    ctx->Message_Block[ctx->Message_Block_Index++] = 0;
                }

                SHA1ProcessMessageBlock(ctx);

                while (ctx->Message_Block_Index < 56) {
                    ctx->Message_Block[ctx->Message_Block_Index++] = 0;
                }
            }
            else
            {
                ctx->Message_Block[ctx->Message_Block_Index++] = 0x80;
                while(ctx->Message_Block_Index < 56) {

                    ctx->Message_Block[ctx->Message_Block_Index++] = 0;
                }
            }

            /*
             *  Store the message length as the last 8 octets
             */
            ctx->Message_Block[56] = (uint8_t)(ctx->Length_High >> 24);
            ctx->Message_Block[57] = (uint8_t)(ctx->Length_High >> 16);
            ctx->Message_Block[58] = (uint8_t)(ctx->Length_High >> 8);
            ctx->Message_Block[59] = (uint8_t)(ctx->Length_High);
            ctx->Message_Block[60] = (uint8_t)(ctx->Length_Low >> 24);
            ctx->Message_Block[61] = (uint8_t)(ctx->Length_Low >> 16);
            ctx->Message_Block[62] = (uint8_t)(ctx->Length_Low >> 8);
            ctx->Message_Block[63] = (uint8_t)(ctx->Length_Low);
            SHA1ProcessMessageBlock(ctx);
        }

        /**
         * Return the 160-bit message digest into the user's array
         */
        inline void SHA1_Final(uint8_t *digest, SHA1_CTX *ctx) {
            int i;

            SHA1PadMessage(ctx);
            memset(ctx->Message_Block, 0, 64);
            ctx->Length_Low = 0;    /* and clear length */
            ctx->Length_High = 0;

            for  (i = 0; i < SHA1_SIZE; i++) {
                digest[i] = (uint8_t)(ctx->Intermediate_Hash[i>>2] >> 8 * ( 3 - ( i & 0x03 ) ));
            }
        }
        
        /*
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         *   BEGIN MD5 CODE
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         *  
         */

        #define MD5_SIZE    16

        #define S11 7
        #define S12 12
        #define S13 17
        #define S14 22
        #define S21 5
        #define S22 9
        #define S23 14
        #define S24 20
        #define S31 4
        #define S32 11
        #define S33 16
        #define S34 23
        #define S41 6
        #define S42 10
        #define S43 15
        #define S44 21

        /* F, G, H and I are basic MD5 functions.
         */
        #define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
        #define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
        #define H(x, y, z) ((x) ^ (y) ^ (z))
        #define I(x, y, z) ((y) ^ ((x) | (~z)))

        /* ROTATE_LEFT rotates x left n bits.  */
        #define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

        /* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
           Rotation is separate from addition to prevent recomputation.  */
        #define FF(a, b, c, d, x, s, ac) { \
            (a) += F ((b), (c), (d)) + (x) + (uint32_t)(ac); \
            (a) = ROTATE_LEFT ((a), (s)); \
            (a) += (b); \
          }
        #define GG(a, b, c, d, x, s, ac) { \
            (a) += G ((b), (c), (d)) + (x) + (uint32_t)(ac); \
            (a) = ROTATE_LEFT ((a), (s)); \
            (a) += (b); \
          }
        #define HH(a, b, c, d, x, s, ac) { \
            (a) += H ((b), (c), (d)) + (x) + (uint32_t)(ac); \
            (a) = ROTATE_LEFT ((a), (s)); \
            (a) += (b); \
          }
        #define II(a, b, c, d, x, s, ac) { \
            (a) += I ((b), (c), (d)) + (x) + (uint32_t)(ac); \
            (a) = ROTATE_LEFT ((a), (s)); \
            (a) += (b); \
          }


        typedef struct 
        {
          uint32_t state[4];        /* state (ABCD) */
          uint32_t count[2];        /* number of bits, modulo 2^64 (lsb first) */
          uint8_t buffer[64];       /* input buffer */
        } MD5_CTX;

        static const uint8_t PADDING[64] = 
        {
            0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };

        /**
         * MD5 initialization - begins an MD5 operation, writing a new ctx.
         */
        inline void MD5_Init(MD5_CTX *ctx) {
            ctx->count[0] = ctx->count[1] = 0;

            /* Load magic initialization constants.
             */
            ctx->state[0] = 0x67452301;
            ctx->state[1] = 0xefcdab89;
            ctx->state[2] = 0x98badcfe;
            ctx->state[3] = 0x10325476;
        }

        /**
         * Encodes input (uint32_t) into output (uint8_t). Assumes len is
         *   a multiple of 4.
         */
        inline void Encode(uint8_t *output, uint32_t *input, uint32_t len) {
            uint32_t i, j;

            for (i = 0, j = 0; j < len; i++, j += 4) {
                output[j] = (uint8_t)(input[i] & 0xff);
                output[j+1] = (uint8_t)((input[i] >> 8) & 0xff);
                output[j+2] = (uint8_t)((input[i] >> 16) & 0xff);
                output[j+3] = (uint8_t)((input[i] >> 24) & 0xff);
            }
        }

        /**
         *  Decodes input (uint8_t) into output (uint32_t). Assumes len is
         *   a multiple of 4.
         */
        inline void Decode(uint32_t *output, const uint8_t *input, uint32_t len) {
            uint32_t i, j;

            for (i = 0, j = 0; j < len; i++, j += 4)
                output[i] = ((uint32_t)input[j]) | (((uint32_t)input[j+1]) << 8) |
                    (((uint32_t)input[j+2]) << 16) | (((uint32_t)input[j+3]) << 24);
        }

        /**
         * MD5 basic transformation. Transforms state based on block.
         */
        inline void MD5Transform(uint32_t state[4], const uint8_t block[64]) {
            uint32_t a = state[0], b = state[1], c = state[2], 
                     d = state[3], x[MD5_SIZE];

            Decode(x, block, 64);

            /* Round 1 */
            FF (a, b, c, d, x[ 0], S11, 0xd76aa478); /* 1 */
            FF (d, a, b, c, x[ 1], S12, 0xe8c7b756); /* 2 */
            FF (c, d, a, b, x[ 2], S13, 0x242070db); /* 3 */
            FF (b, c, d, a, x[ 3], S14, 0xc1bdceee); /* 4 */
            FF (a, b, c, d, x[ 4], S11, 0xf57c0faf); /* 5 */
            FF (d, a, b, c, x[ 5], S12, 0x4787c62a); /* 6 */
            FF (c, d, a, b, x[ 6], S13, 0xa8304613); /* 7 */
            FF (b, c, d, a, x[ 7], S14, 0xfd469501); /* 8 */
            FF (a, b, c, d, x[ 8], S11, 0x698098d8); /* 9 */
            FF (d, a, b, c, x[ 9], S12, 0x8b44f7af); /* 10 */
            FF (c, d, a, b, x[10], S13, 0xffff5bb1); /* 11 */
            FF (b, c, d, a, x[11], S14, 0x895cd7be); /* 12 */
            FF (a, b, c, d, x[12], S11, 0x6b901122); /* 13 */
            FF (d, a, b, c, x[13], S12, 0xfd987193); /* 14 */
            FF (c, d, a, b, x[14], S13, 0xa679438e); /* 15 */
            FF (b, c, d, a, x[15], S14, 0x49b40821); /* 16 */

            /* Round 2 */
            GG (a, b, c, d, x[ 1], S21, 0xf61e2562); /* 17 */
            GG (d, a, b, c, x[ 6], S22, 0xc040b340); /* 18 */
            GG (c, d, a, b, x[11], S23, 0x265e5a51); /* 19 */
            GG (b, c, d, a, x[ 0], S24, 0xe9b6c7aa); /* 20 */
            GG (a, b, c, d, x[ 5], S21, 0xd62f105d); /* 21 */
            GG (d, a, b, c, x[10], S22,  0x2441453); /* 22 */
            GG (c, d, a, b, x[15], S23, 0xd8a1e681); /* 23 */
            GG (b, c, d, a, x[ 4], S24, 0xe7d3fbc8); /* 24 */
            GG (a, b, c, d, x[ 9], S21, 0x21e1cde6); /* 25 */
            GG (d, a, b, c, x[14], S22, 0xc33707d6); /* 26 */
            GG (c, d, a, b, x[ 3], S23, 0xf4d50d87); /* 27 */
            GG (b, c, d, a, x[ 8], S24, 0x455a14ed); /* 28 */
            GG (a, b, c, d, x[13], S21, 0xa9e3e905); /* 29 */
            GG (d, a, b, c, x[ 2], S22, 0xfcefa3f8); /* 30 */
            GG (c, d, a, b, x[ 7], S23, 0x676f02d9); /* 31 */
            GG (b, c, d, a, x[12], S24, 0x8d2a4c8a); /* 32 */

            /* Round 3 */
            HH (a, b, c, d, x[ 5], S31, 0xfffa3942); /* 33 */
            HH (d, a, b, c, x[ 8], S32, 0x8771f681); /* 34 */
            HH (c, d, a, b, x[11], S33, 0x6d9d6122); /* 35 */
            HH (b, c, d, a, x[14], S34, 0xfde5380c); /* 36 */
            HH (a, b, c, d, x[ 1], S31, 0xa4beea44); /* 37 */
            HH (d, a, b, c, x[ 4], S32, 0x4bdecfa9); /* 38 */
            HH (c, d, a, b, x[ 7], S33, 0xf6bb4b60); /* 39 */
            HH (b, c, d, a, x[10], S34, 0xbebfbc70); /* 40 */
            HH (a, b, c, d, x[13], S31, 0x289b7ec6); /* 41 */
            HH (d, a, b, c, x[ 0], S32, 0xeaa127fa); /* 42 */
            HH (c, d, a, b, x[ 3], S33, 0xd4ef3085); /* 43 */
            HH (b, c, d, a, x[ 6], S34,  0x4881d05); /* 44 */
            HH (a, b, c, d, x[ 9], S31, 0xd9d4d039); /* 45 */
            HH (d, a, b, c, x[12], S32, 0xe6db99e5); /* 46 */
            HH (c, d, a, b, x[15], S33, 0x1fa27cf8); /* 47 */
            HH (b, c, d, a, x[ 2], S34, 0xc4ac5665); /* 48 */

            /* Round 4 */
            II (a, b, c, d, x[ 0], S41, 0xf4292244); /* 49 */
            II (d, a, b, c, x[ 7], S42, 0x432aff97); /* 50 */
            II (c, d, a, b, x[14], S43, 0xab9423a7); /* 51 */
            II (b, c, d, a, x[ 5], S44, 0xfc93a039); /* 52 */
            II (a, b, c, d, x[12], S41, 0x655b59c3); /* 53 */
            II (d, a, b, c, x[ 3], S42, 0x8f0ccc92); /* 54 */
            II (c, d, a, b, x[10], S43, 0xffeff47d); /* 55 */
            II (b, c, d, a, x[ 1], S44, 0x85845dd1); /* 56 */
            II (a, b, c, d, x[ 8], S41, 0x6fa87e4f); /* 57 */
            II (d, a, b, c, x[15], S42, 0xfe2ce6e0); /* 58 */
            II (c, d, a, b, x[ 6], S43, 0xa3014314); /* 59 */
            II (b, c, d, a, x[13], S44, 0x4e0811a1); /* 60 */
            II (a, b, c, d, x[ 4], S41, 0xf7537e82); /* 61 */
            II (d, a, b, c, x[11], S42, 0xbd3af235); /* 62 */
            II (c, d, a, b, x[ 2], S43, 0x2ad7d2bb); /* 63 */
            II (b, c, d, a, x[ 9], S44, 0xeb86d391); /* 64 */

            state[0] += a;
            state[1] += b;
            state[2] += c;
            state[3] += d;
        }

        /**
         * Accepts an array of octets as the next portion of the message.
         */
        inline void MD5_Update(MD5_CTX *ctx, const uint8_t * msg, int len) {
            uint32_t x;
            int i, partLen;

            /* Compute number of bytes mod 64 */
            x = (uint32_t)((ctx->count[0] >> 3) & 0x3F);

            /* Update number of bits */
            if ((ctx->count[0] += ((uint32_t)len << 3)) < ((uint32_t)len << 3))
                ctx->count[1]++;
            ctx->count[1] += ((uint32_t)len >> 29);

            partLen = 64 - x;

            /* Transform as many times as possible.  */
            if (len >= partLen) {
                memcpy(&ctx->buffer[x], msg, partLen);
                MD5Transform(ctx->state, ctx->buffer);

                for (i = partLen; i + 63 < len; i += 64)
                    MD5Transform(ctx->state, &msg[i]);

                x = 0;
            }
            else
                i = 0;

            /* Buffer remaining input */
            memcpy(&ctx->buffer[x], &msg[i], len-i);
        }

        /**
         * Return the 128-bit message digest into the user's array
         */
        inline void MD5_Final(uint8_t *digest, MD5_CTX *ctx) {
            uint8_t bits[8];
            uint32_t x, padLen;

            /* Save number of bits */
            Encode(bits, ctx->count, 8);

            /* Pad out to 56 mod 64.
             */
            x = (uint32_t)((ctx->count[0] >> 3) & 0x3f);
            padLen = (x < 56) ? (56 - x) : (120 - x);
            MD5_Update(ctx, PADDING, padLen);

            /* Append length (before padding) */
            MD5_Update(ctx, bits, 8);

            /* Store state in digest */
            Encode(digest, ctx->state, MD5_SIZE);
        }
        
        /*
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         *   BEGIN RSA CODE
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         *  
         */
        typedef struct {
            bigint *m;              /* modulus */
            bigint *e;              /* public exponent */
            bigint *d;              /* private exponent */
            int num_octets;
            BI_CTX *bi_ctx;
        } RSA_CTX;

        /**
         * Free up any RSA context resources.
         */
        inline void RSA_free(RSA_CTX *rsa_ctx) {
            BI_CTX *bi_ctx;
            if (rsa_ctx == NULL)                /* deal with ptrs that are null */
                return;

            bi_ctx = rsa_ctx->bi_ctx;

            bi_depermanent(rsa_ctx->e);
            bi_free(bi_ctx, rsa_ctx->e);
            bi_free_mod(rsa_ctx->bi_ctx, BIGINT_M_OFFSET);

            if (rsa_ctx->d) {
                bi_depermanent(rsa_ctx->d);
                bi_free(bi_ctx, rsa_ctx->d);
            }

            bi_terminate(bi_ctx);
            free(rsa_ctx);
        }

        inline void RSA_pub_key_new(RSA_CTX **ctx, 
                const uint8_t *modulus, int mod_len,
                const uint8_t *pub_exp, int pub_len) {
            RSA_CTX *rsa_ctx;
            BI_CTX *bi_ctx;

            if (*ctx)   /* if we load multiple certs, dump the old one */
                RSA_free(*ctx);

            bi_ctx = bi_initialize();
            *ctx = (RSA_CTX *)calloc(1, sizeof(RSA_CTX));
            rsa_ctx = *ctx;
            rsa_ctx->bi_ctx = bi_ctx;

            // num_octets is the largest multiple of 16 still smaller than mod_len
            rsa_ctx->num_octets = (mod_len & 0xFFF0);
            rsa_ctx->m = bi_import(bi_ctx, modulus, mod_len);
            bi_set_mod(bi_ctx, rsa_ctx->m, BIGINT_M_OFFSET);
            rsa_ctx->e = bi_import(bi_ctx, pub_exp, pub_len);
            bi_permanent(rsa_ctx->e);
        }

        inline void RSA_priv_key_new(RSA_CTX **ctx, 
                const uint8_t *modulus, int mod_len,
                const uint8_t *pub_exp, int pub_len,
                const uint8_t *priv_exp, int priv_len
            ) {
            RSA_CTX *rsa_ctx;
            BI_CTX *bi_ctx;
            RSA_pub_key_new(ctx, modulus, mod_len, pub_exp, pub_len);
            rsa_ctx = *ctx;
            bi_ctx = rsa_ctx->bi_ctx;
            rsa_ctx->d = bi_import(bi_ctx, priv_exp, priv_len);
            bi_permanent(rsa_ctx->d);
        }

        /**
         * Performs c = m^e mod n
         */
        inline bigint *RSA_public(const RSA_CTX * c, bigint *bi_msg) {
            c->bi_ctx->mod_offset = BIGINT_M_OFFSET;
            return bi_mod_power(c->bi_ctx, bi_msg, c->e);
        }

        /**
         * Performs m = c^d mod n
         */
        inline bigint *RSA_private(const RSA_CTX *c, bigint *bi_msg) {
            BI_CTX *ctx = c->bi_ctx;
            ctx->mod_offset = BIGINT_M_OFFSET;
            return bi_mod_power(ctx, bi_msg, c->d);
        }

        /**
         * Use PKCS1.5 for encryption/signing.
         * see http://www.rsasecurity.com/rsalabs/node.asp?id=2125
         */
        inline int RSA_encrypt(const RSA_CTX *ctx, const uint8_t *in_data, uint16_t in_len, 
                uint8_t *out_data, int is_signing) {
            int byte_size = ctx->num_octets;
            int num_pads_needed = byte_size-in_len-3;
            bigint *dat_bi, *encrypt_bi;

            /* note: in_len must be <= byte_size - 11 */
            out_data[0] = 0;     /* ensure encryption block is < modulus */

            if (is_signing) {
                out_data[1] = 1;        /* PKCS1.5 signing pads with "0xff"'s */
                memset(&out_data[2], 0xff, num_pads_needed);
            }
            else /* randomize the encryption padding with non-zero bytes */   
            {
                out_data[1] = 2;
                get_random_NZ(num_pads_needed, &out_data[2]);
            }

            out_data[2+num_pads_needed] = 0;
            memcpy(&out_data[3+num_pads_needed], in_data, in_len);

            /* now encrypt it */
            dat_bi = bi_import(ctx->bi_ctx, out_data, byte_size);
            encrypt_bi = is_signing ? RSA_private(ctx, dat_bi) : 
                                      RSA_public(ctx, dat_bi);
            bi_export(ctx->bi_ctx, encrypt_bi, out_data, byte_size);

            /* save a few bytes of memory */
            bi_clear_cache(ctx->bi_ctx);
            return byte_size;
        }

        /**
         * @brief Use PKCS1.5 for decryption/verification.
         * @param ctx [in] The context
         * @param in_data [in] The data to encrypt (must be < modulus size-11)
         * @param out_data [out] The encrypted data.
         * @param is_decryption [in] Decryption or verify operation.
         * @return  The number of bytes that were originally encrypted. -1 on error.
         * @see http://www.rsasecurity.com/rsalabs/node.asp?id=2125
         */
        inline int RSA_decrypt(const RSA_CTX *ctx, const uint8_t *in_data, 
                                    uint8_t *out_data, int is_decryption) {
            const int byte_size = ctx->num_octets;
            int i, size;
            bigint *decrypted_bi, *dat_bi;
            uint8_t *block = (uint8_t *) malloc(byte_size);

            memset(out_data, 0, byte_size); /* initialise */

            /* decrypt */
            dat_bi = bi_import(ctx->bi_ctx, in_data, byte_size);
            decrypted_bi = is_decryption ?  /* decrypt or verify? */
                    RSA_private(ctx, dat_bi) : RSA_public(ctx, dat_bi);

            /* convert to a normal block */
            bi_export(ctx->bi_ctx, decrypted_bi, block, byte_size);

            i = 10; /* start at the first possible non-padded byte */

            if (is_decryption == 0) /* PKCS1.5 signing pads with "0xff"s */
            {
                while (block[i++] == 0xff && i < byte_size);

                if (block[i-2] != 0xff)
                    i = byte_size;     /*ensure size is 0 */   
            }
            else                    /* PKCS1.5 encryption padding is random */
            {
                while (block[i++] && i < byte_size);
            }
            size = byte_size - i;

            /* get only the bit we want */
            if (size > 0)
                memcpy(out_data, &block[i], size);
            
            return size ? size : -1;
        }
    }

    /*
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    *   Begin API code
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    * 
    *  
    */
    
    using namespace Crypto;

    inline size_t hashMD5(const uint8_t* in, size_t in_size, uint8_t* out) {
        MD5_CTX ctx;

        if (out) {
            MD5_Init(&ctx);
            MD5_Update(&ctx, (uint8_t*) in, (int)in_size);
            MD5_Final((uint8_t*) out, &ctx);
        }

        return MD5_SIZE;
    }

    inline size_t hashSHA1(const uint8_t* in, size_t in_size, uint8_t* out) {
        SHA1_CTX ctx;

        if (out) {
            SHA1_Init(&ctx);
            SHA1_Update(&ctx, in, (int)in_size);
            SHA1_Final(out, &ctx);
        } else
			return 0;

        return SHA1_SIZE;
    }

	inline size_t hashSHA1(iStream* stream, uint8_t* out) {
		SHA1_CTX ctx;
		size_t size,def_chunk = 4096;
		Buf buf(def_chunk);

		if (stream && out) {
			size = stream->size();
			SHA1_Init(&ctx);

			while (size){
				size_t chunk = min(size, def_chunk);
				if (chunk != stream->read(buf.ptr(), chunk))
					return 0;

				buf.resize(chunk);
				size -= chunk;
				SHA1_Update(&ctx, (uint8_t*) buf.ptr(), (int)chunk);			
			}
						
			SHA1_Final(out, &ctx);
		} else
			return 0;

		return SHA1_SIZE;
	}

    inline size_t encryptRSA(uint8_t* in, size_t in_size, uint8_t* out, uint8_t* mod,
                        size_t mod_size, uint8_t* exp, size_t exp_size,
                        bool prv = true) {
        size_t out_size = 0;
        size_t data_size = mod_size & 0xFFF0;
     
        RSA_CTX* ctx = 0;
        uint8_t* data = 0;

        if (out) {
            ctx = 0;

            if (prv) {
                RSA_priv_key_new(&ctx, 
                    mod, (int)mod_size,
                    0, 0,
                    exp, (int)exp_size
                );
            } else {
                RSA_priv_key_new(&ctx, 
                    mod, (int)mod_size,
                    exp, (int)exp_size,
                    0, 0
                );
            }
            
            data = (uint8_t*)g_sys->alloc(data_size);
        }

        while (in_size > 0) {
            size_t size = min(in_size, data_size - 11);
            
            if (out) {
                RSA_encrypt(ctx, in, (uint16_t)size, data, prv);

                memcpy(out, data, data_size);

                in += size;
                out += data_size;
            }

            in_size -= size;
            out_size += data_size;
        }

        if (out) {
            g_sys->free(data);
            RSA_free(ctx);
        }

        return out_size;
    }

    inline size_t decryptRSA(const uint8_t* in, size_t in_size, uint8_t* out, const uint8_t* mod,
                        size_t mod_size, const uint8_t* exp, size_t exp_size,
                        bool pub = true) {
//        o3_assert(out);

        size_t out_size = 0;
        size_t data_size = mod_size & 0xFFF0;
        
        RSA_CTX* ctx = 0;

        if (pub) {
            RSA_priv_key_new(&ctx, 
                mod, (int)mod_size,
                exp, (int)exp_size,
                0, 0
            );
        } else {
            RSA_priv_key_new(&ctx, 
               (uint8_t*) mod, (int)mod_size,
                0, 0,
                (uint8_t*) exp, (int)exp_size
            );			
        }

        uint8_t* data = (uint8_t*)g_sys->alloc(data_size);

        while (in_size >= data_size) {
            memcpy(data, in, data_size);

            size_t size = RSA_decrypt(ctx, data, out, !pub);

            in += data_size;
            out += size;

            in_size -= data_size;
            out_size += size;
        }

        g_sys->free(data);
        RSA_free(ctx);

        return out_size;
    }
}

// TODO: get rid of these crappy macros:

#undef V1
#undef V2
#undef U
#undef Q
#undef V1
#undef V2
#undef S11 
#undef S12 
#undef S13 
#undef S14 
#undef S21
#undef S22 
#undef S23 
#undef S24 
#undef S31
#undef S32 
#undef S33 
#undef S34 
#undef S41
#undef S42 
#undef S43 
#undef S44 
#undef F
#undef G
#undef H
#undef I
#undef FF
#undef GG
#undef HH
#undef II
#endif // O3_CRYPTO_H

/*
 * Copyright (C) 2010 Ajax.org BV
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this library; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */
#ifndef O3_TEST_T_VEC_H
#define O3_TEST_T_VEC_H

namespace o3 {

struct X {
    static unsigned c;
    static unsigned d;
    int x;

    X(int x) : x(x)
    {
        ++c;
    }

    X(const X& that) : x(that.x)
    {
        ++c;
    }

    ~X()
    {
        ++d;
    }
};

unsigned X::c;
unsigned X::d;

void test_tVec()
{
    o3_log("Testing explicit tVec::tVec(size_t, iAlloc*)\n"); 
    {
        tVec<X> vec(100);

        o3_assert(vec.capacity() == 128);
        o3_assert(vec.size() == 0);
    }

    o3_log("Testing tVec::tVec(const tVec&)\n");
    {
        X::c = 0;
        X::d = 0;
        tVec<X> vec;

        vec.append(X(10), 10);
        o3_assert(X::c == 11);
        o3_assert(X::d == 1);
        {
            const tVec<X> vec1 = vec;

            o3_assert(X::c == 11);
            o3_assert(X::d == 1);
            o3_assert(vec1.capacity() == vec.capacity());
            o3_assert(vec1.size() == vec.size());
            for (size_t i = 0; i < vec.size(); ++i)
                o3_assert(vec1[i].x == ((const tVec<X>&) vec)[i].x);
        }
        o3_assert(X::c == 11);
        o3_assert(X::d == 1);
    }

    o3_log("Testing T* tVec::ptr()\n");
    {
        X::c = 0;
        X::d = 0;
        tVec<X> vec;

        vec.append(X(10), 10);
        {
            tVec<X> vec1 = vec;

            vec1.ptr();
            o3_assert(X::c == 21);
            o3_assert(X::d == 1);
            o3_assert(vec1.capacity() == vec.capacity());
            o3_assert(vec1.size() == vec.size());
            for (size_t i = 0; i < vec.size(); ++i)
                o3_assert(vec1[i].x == vec[i].x);
        }
        o3_assert(X::c == 21);
        o3_assert(X::d == 11);
    }

    o3_log("Testing void tVec::insert(size_t, const T&, size_t)\n");
    {
        X::c = 0;
        X::d = 0;
        tVec<X> vec;

        vec.append(X(42), 20);
        vec.insert(10, X(21), 10);
        o3_assert(X::c == 32);
        o3_assert(X::d == 2);
        o3_assert(vec.capacity() == 32);
        o3_assert(vec.size() == 30);
        for (size_t i = 0; i < vec.size(); ++i) 
            if (i < 10 || i >= 20)
                o3_assert(vec[i].x == 42);
            else
                o3_assert(vec[i].x == 21);
    }

    o3_log("Testing void tVec::insert(size_t, const T*, size_t)\n");
    {
        X::c = 0;
        X::d = 0;
        tVec<X> vec;

        vec.append(X(42), 20);
        {
            tVec<X> vec1;

            vec1.append(X(21), 10);
            vec.insert(10, vec1.ptr(), vec1.size());
        }
        o3_assert(X::c == 42);
        o3_assert(X::d == 12);
        o3_assert(vec.capacity() == 32);
        o3_assert(vec.size() == 30);
        for (size_t i = 0; i < vec.size(); ++i) 
            if (i < 10 || i >= 20)
                o3_assert(vec[i].x == 42);
            else
                o3_assert(vec[i].x == 21);
    }

    o3_log("Testing void tVec::append(const T&, size_t)\n");
    {
        X::c = 0;
        X::d = 0;
        tVec<X> vec;

        vec.append(X(42), 20);
        vec.append(X(21), 10);
        o3_assert(X::c == 32);
        o3_assert(X::d == 2);
        o3_assert(vec.capacity() == 32);
        o3_assert(vec.size() == 30);
        for (size_t i = 0; i < vec.size(); ++i) 
            if (i < 20)
                o3_assert(vec[i].x == 42);
            else
                o3_assert(vec[i].x == 21);
    }

    o3_log("Testing void tVec::append(const T*, size_t)\n");
    {
        X::c = 0;
        X::d = 0;
        tVec<X> vec;

        vec.append(X(42), 20);
        {
            tVec<X> vec1;

            vec1.append(X(21), 10);
            vec.append(vec1.ptr(), vec1.size());
        }
        o3_assert(X::c == 42);
        o3_assert(X::d == 12);
        o3_assert(vec.capacity() == 32);
        o3_assert(vec.size() == 30);
        for (size_t i = 0; i < vec.size(); ++i) 
            if (i < 20)
                o3_assert(vec[i].x == 42);
            else
                o3_assert(vec[i].x == 21);
    }

    o3_log("Testing void tVec::remove(size_t, size_t)\n");
    {
        X::c = 0;
        X::d = 0;
        tVec<X> vec;

        vec.append(X(42), 20);
        vec.insert(10, X(21), 10);
        vec.remove(5, 20);
        o3_assert(X::c == 32);
        o3_assert(X::d == 22);
        o3_assert(vec.capacity() == 32);
        o3_assert(vec.size() == 10);
        for (size_t i = 0; i < vec.size(); ++i)
            o3_assert(vec[i].x == 42);
    }
}

}

#endif // O3_TEST_T_VEC_H

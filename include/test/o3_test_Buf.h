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
#ifndef O3_TEST_BUF_H
#define O3_TEST_BUF_H

namespace o3 {

inline void test_Buf()
{
    const char STR[] = "The quick brown fox jumps over the lazy dog";
    const wchar_t WSTR[] = L"The quick brown fox jumps over the lazy dog";
    void* ptr = memCopy(memAlloc(sizeof(STR)), STR, sizeof(STR));
    char x[] = { 'C', '3', 'P', 'O' };
    
    o3_log("Testing static Buf Buf::fromHex(const char*, iAlloc*)\n");
    {
        Buf buf = Buf::fromHex("54 68 65 20 71 75 69 63 6B 20 62"
                               "72 6F 77 6E 20 66 6F 78 20 6A 75"
                               "6D 70 73 20 6F 76 65 72 20 74 68"
                               "65 20 6C 61 7A 79 20 64 6F 67 00");

        o3_assert(buf.capacity() == 64);
        o3_assert(buf.size() == sizeof(STR));
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR)));
    }

    o3_log("Testing static Buf Buf::fromHex(const wchar_t*, iAlloc*)\n");
    {
		/*
		//!TODO FIX THIS!
        Buf buf = Buf::fromHex(L"54 68 65 20 71 75 69 63 6B 20 62"
                               L"72 6F 77 6E 20 66 6F 78 20 6A 75"
                               L"6D 70 73 20 6F 76 65 72 20 74 68"
                               L"65 20 6C 61 7A 79 20 64 6F 67 00");

        o3_assert(buf.capacity() == 64);
        o3_assert(buf.size() == sizeof(STR));
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR)));*/
    }

    o3_log("Testing static Buf Buf::fromBase64(const char*, iAlloc*)\n");
    {
        Buf buf = Buf::fromBase64("VAAAAGgAAABlAAAAIAAAAHEAAAB1AAAAaQ"
                                  "AAAGMAAABrAAAAIAAAAGIAAAByAAAAbwAA"
                                  "AHcAAABuAAAAIAAAAGYAAABvAAAAeAAAAC"
                                  "AAAABqAAAAdQAAAG0AAABwAAAAcwAAACAA"
                                  "AABvAAAAdgAAAGUAAAByAAAAIAAAAHQAAA"
                                  "BoAAAAZQAAACAAAABsAAAAYQAAAHoAAAB5"
                                  "AAAAIAAAAGQAAABvAAAAZwAAAAAAAAA=");

        o3_assert(buf.capacity() == 256);
#ifndef O3_WIN32
		o3_assert(buf.size() == sizeof(WSTR));
        o3_assert(memEquals(buf.ptr(), WSTR, sizeof(WSTR)));
#endif
	}

    o3_log("Testing static Buf Buf::fromBase64(const wchar_t*, iAlloc*)\n");
    {
        Buf buf = Buf::fromBase64(L"VAAAAGgAAABlAAAAIAAAAHEAAAB1AAAAaQ"
                                  L"AAAGMAAABrAAAAIAAAAGIAAAByAAAAbwAA"
                                  L"AHcAAABuAAAAIAAAAGYAAABvAAAAeAAAAC"
                                  L"AAAABqAAAAdQAAAG0AAABwAAAAcwAAACAA"
                                  L"AABvAAAAdgAAAGUAAAByAAAAIAAAAHQAAA"
                                  L"BoAAAAZQAAACAAAABsAAAAYQAAAHoAAAB5"
                                  L"AAAAIAAAAGQAAABvAAAAZwAAAAAAAAA=");

        o3_assert(buf.capacity() == 256);
#ifndef O3_WIN32
		o3_assert(buf.size() == sizeof(WSTR));
        o3_assert(memEquals(buf.ptr(), WSTR, sizeof(WSTR)));
#endif
    }

    o3_log("Testing explicit Buf::Buf(size_t, iAlloc*)\n");
    {
        {
            Buf buf;

            o3_assert(buf.capacity() == O3_AUTO_CAPACITY);
            o3_assert(buf.size() == 0);
        } {
            Buf buf(100);

            o3_assert(buf.capacity() == 128);
            o3_assert(buf.size() == 0);
        }
    }

    o3_log("Testing Buf::Buf(void*, size_t)\n");
    {
        Buf buf(ptr, sizeof(STR)); 

        o3_assert(buf.capacity() == sizeof(STR));
        o3_assert(buf.size() == sizeof(STR));
        o3_assert(buf.ptr() == ptr);
    }

    o3_log("Testing Buf::Buf(iStream*, iAlloc*)\n");
    {
        // TODO
    }

    o3_log("Testing Buf::Buf(iStream*, size_t, iAlloc*)\n");
    {
        // TODO
    }

    o3_log("Testing Buf::Buf(const Buf&)\n");
    {
        {
            Buf buf;

            buf.append("TEST123", 8);
            {
                Buf buf1 = buf;

                o3_assert(buf.unique());
                o3_assert(buf1.unique());
                o3_assert(buf1.capacity() == buf.capacity());
                o3_assert(buf1.size() == buf.size());
                o3_assert(buf1.ptr() != buf.ptr());
                o3_assert(memEquals(buf1.ptr(), buf.ptr(), buf.size()));
            }
        } {
            Buf buf;

            buf.append(STR, sizeof(STR));
            {        
                Buf buf1 = buf;

                o3_assert(!buf.unique());
                o3_assert(!buf1.unique());
                o3_assert(buf1.capacity() == buf.capacity());
                o3_assert(buf1.size() == buf.size());
                o3_assert(((const Buf&) buf1).ptr() == ((const Buf&) buf).ptr());
            }
            o3_assert(buf.unique());
        } {
            Buf buf(ptr, sizeof(STR));
            Buf buf1 = buf;

            o3_assert(buf.unique());
            o3_assert(buf1.unique());
            o3_assert(buf1.capacity() == buf.capacity());
            o3_assert(buf1.size() == buf.size());
            o3_assert(buf1.ptr() == buf.ptr());
        }
    }

    o3_log("Testing bool Buf::operator==(const Buf&)\n");
    {
        Buf buf;

        buf.append(STR, sizeof(STR));
        o3_assert(buf == buf);
        {
            Buf buf1;

            buf1.append(STR, sizeof(STR));
            o3_assert(buf1 == buf);
        } {
            const char STR1[] = "The lazy dog jumps over the quick brown fox";
            Buf buf1;

            buf1.append(STR1, sizeof(STR1));
            o3_assert(!(buf1 == buf));
        } {
            const char STR1[] = "The quick brown fox";
            Buf buf1;

            buf1.append(STR1, sizeof(STR1));
            o3_assert(!(buf1 == buf));
        } 
    }

    o3_log("Testing bool Buf::operator<(const Buf&)\n");
    {
        Buf buf;

        buf.append(WSTR, sizeof(WSTR));
        o3_assert(!(buf < buf));
        {
            Buf buf1;

            buf1.append(WSTR, sizeof(WSTR));
            o3_assert(!(buf1 < buf));
            o3_assert(!(buf < buf1));
        } {
            const wchar_t WSTR1[] = L"The lazy dog"
                                    L"jumps over the quick brown fox";
            Buf buf1;

            buf1.append(WSTR1, sizeof(WSTR1));
            o3_assert(buf1 < buf);
            o3_assert(!(buf < buf1));
        } {
            const wchar_t WSTR1[] = L"The quick brown fox";
            Buf buf1;

            buf1.append(WSTR1, sizeof(WSTR1));
            o3_assert(buf1 < buf);
            o3_assert(!(buf < buf1));
        }
    }

    o3_log("Testing size_t Buf::find(size_t, const void*, size_t) const\n");
    {
        const char STR1[] = "jumps over";
        const char STR2[] = "jumps under";
        Buf buf;

        buf.append(STR, sizeof(STR));
        o3_assert(buf.find(10, STR1, sizeof(STR1) - 1) == 20);       
        o3_assert(buf.find(30, STR1, sizeof(STR1) - 1) == NOT_FOUND);
        o3_assert(buf.find(20, STR2, sizeof(STR2) - 1) == NOT_FOUND);       
    }

    o3_log("Testing void Buf::reserve(size_t)\n");
    {
        Buf buf;

        buf.append(STR, sizeof(STR));
        buf.reserve(50);
        o3_assert(buf.capacity() == 64);
        o3_assert(buf.size() == sizeof(STR));
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR)));
        buf.reserve(64);
        o3_assert(buf.capacity() == 64);
        o3_assert(buf.size() == sizeof(STR));
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR)));
        buf.reserve(100);
        o3_assert(buf.capacity() == 128);
        o3_assert(buf.size() == sizeof(STR));
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR)));
    }

    o3_log("Testing void Buf::resize(size_t)\n");
    {
        Buf buf;

        buf.append(STR, sizeof(STR));
        buf.resize(sizeof(STR) / 2);
        o3_assert(buf.capacity() == 64);
        o3_assert(buf.size() == sizeof(STR) / 2);
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR) / 2));
        buf.resize(sizeof(STR));
        o3_assert(buf.capacity() == 64);
        o3_assert(buf.size() == sizeof(STR));
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR) / 2));
        buf.resize(sizeof(STR) * 2);
        o3_assert(buf.capacity() == 128);
        o3_assert(buf.size() == sizeof(STR) * 2);
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR) / 2));
    }

    o3_log("Testing void* Buf::ptr()\n");
    {
        Buf buf;

        buf.append(STR, sizeof(STR));
        {
            Buf buf1 = buf;

            o3_assert(buf1.ptr() != buf.ptr());
            o3_assert(memEquals(buf1.ptr(), buf.ptr(), buf.size()));
            o3_assert(buf1.unique());
        }
        o3_assert(buf.unique());
    }

    o3_log("Testing void* Buf::set(size_t, const T&, size_t)\n");
    {
        const char STR1[] = "The quick brown fC3POC3POC3ver the lazy dog";
        Buf buf;

        buf.append(STR, sizeof(STR));
        buf.set(17, x, 10);
        o3_assert(memEquals(buf.ptr(), STR1, sizeof(STR1)));
    }

    o3_log("Testing void* Buf::copy(size_t, const void*, size_t)\n");
    {
        const char STR1[] = "The quick brown fox walks over the lazy dog";
        const char STR2[] = "walks over";
        Buf buf;

        buf.append(STR, sizeof(STR));
        buf.copy(20, STR2, sizeof(STR2) - 1);
        o3_assert(memEquals(buf.ptr(), STR1, sizeof(STR1)));
    }

    o3_log("Testing void* Buf::move(size_t, size_t, size_t)\n");
    {
        const char STR1[] = "The quick brown dog jumps over the lazy dog";
        Buf buf;

        buf.append(STR, sizeof(STR));
        buf.move(16, 40, 3);
        o3_assert(memEquals(buf.ptr(), STR1, sizeof(STR1)));
    }

    o3_log("Testing void shift(size_t, size_t)\n");
    {
        Buf buf;

        buf.append(STR, sizeof(STR));
        buf.shift(20, 10);
        o3_assert(memEquals(buf.ptr(), STR, 20));
        o3_assert(memEquals((uint8_t*) buf.ptr() + 30, STR + 20, 24));
    }

    o3_log("Testing void Buf::insertPattern(size_t, const T&, size_t)\n");
    {
        const char STR1[] = "The quick brown fox the lazy dog";
        const char STR2[] = "The quick brown fox C3POC3POC3the lazy dog";
        Buf buf;

        buf.append(STR1, sizeof(STR1));
        buf.insertPattern(20, x, 10);
        o3_assert(memEquals(buf.ptr(), STR2, sizeof(STR2)));
    }

    o3_log("Testing void Buf::insert(size_t, const void*, size_t)\n");
    {
        const char STR1[] = "The quick brown fox the lazy dog";
        const char STR2[] = "jumps over ";
        Buf buf;

        buf.append(STR1, sizeof(STR1));
        buf.insert(20, STR2, sizeof(STR2) - 1);
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR)));
    }

    o3_log("Testing void Buf::appendPattern(const T&, size_t)\n");
    {
        const char STR1[] = "The quick brown fox ";
        const char STR2[] = "The quick brown fox C3POC3POC3";
        Buf buf;

        buf.append(STR1, sizeof(STR1) - 1);
        buf.appendPattern(x, 10);
        o3_assert(memEquals(buf.ptr(), STR2, sizeof(STR2) - 1));
    }

    o3_log("Testing void Buf::append(const void*, size_t)\n");
    {
        const char STR1[] = "The quick brown fox ";
        const char STR2[] = "jumps over the lazy dog";
        Buf buf;

        buf.append(STR1, sizeof(STR1) - 1);
        buf.append(STR2, sizeof(STR2));
        o3_assert(memEquals(buf.ptr(), STR, sizeof(STR)));
    }

    o3_log("Testing void Buf::remove(size_t, size_t n)\n");
    {
        const char STR1[] = "The quick brown fox the lazy dog";
        Buf buf;

        buf.append(STR, sizeof(STR));
        buf.remove(20, 11);
        o3_assert(memEquals(buf.ptr(), STR1, sizeof(STR1)));
    }

    o3_log("Testing void Buf::replace(size_t, size_t, const T&, size_t n1)\n");
    {
        // TODO
    }

    o3_log("Testing void Buf::replace(size_t, size_t, const void*, size_t)\n");
    {
        // TODO
    }

    free(ptr);
}

}

#endif // O3_TEST_BUF_H


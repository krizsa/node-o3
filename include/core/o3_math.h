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
#ifndef O3_MATH_H
#define O3_MATH_H

namespace o3
{
	static const double pi = 3.1415926535897932384626433832795;
	template<typename T>
	class V2
	{
	public:
		V2()
		{
			x = y = 0;
		};

		template <typename C> V2(C _x, C _y)
		{
			x = (T)_x;
			y = (T)_y;
		};

		
		T x,y;
	
		V2<T> operator + (V2<T> &B)
		{
			return V2<T>(x + B.x, y + B.y);
		};

		V2<T> operator - (V2<T> &B)
		{
			return V2<T>(x - B.x, y - B.y);
		};

		V2<T> operator * (T &B)
		{
			return V2<T>(x * B, y * B);
		};
	};	
	
	template<typename T>
	class M33
	{
	public:
		T M[3][3];
		
		M33()
		{
			setIdentity();
		};
		
		void setIdentity()
		{
			M[0][0] = M[1][1] = M[2][2] = 1.0f;
			M[0][1] = M[0][2] = M[1][0] = 
			M[1][2] = M[2][0] = M[2][1] = 0.0f;
		};

		void setTranslation(T x, T y)
		{
			setIdentity();
			M[2][0] = x;
			M[2][1] = y;
		};

		void setScale(T scalex, T scaley)
		{

			M[0][0] = scalex;
			M[1][1] = scaley;
			M[2][2] = 1.0f;
			M[0][1] = M[0][2] = M[1][0] = 
			M[1][2] = M[2][0] = M[2][1] = 0.0f;
		};

		void setRotation(T theta)
		{
			setIdentity();
			T ct  = cos(theta);
			T st  = sin(theta);

			M[0][0] = ct;
			M[1][1] = ct;
			M[1][0] = -st;
			M[0][1] = st;
		};

		M33<T> Multiply(M33<T> &B)
		{
			M33<T> R;
			R.M[0][0] = M[0][0]*B.M[0][0] + M[1][0]*B.M[0][1] + M[2][0]*B.M[0][2];
			R.M[1][0] = M[0][0]*B.M[1][0] + M[1][0]*B.M[1][1] + M[2][0]*B.M[1][2];
			R.M[2][0] = M[0][0]*B.M[2][0] + M[1][0]*B.M[2][1] + M[2][0]*B.M[2][2];
			
			R.M[0][1] = M[0][1]*B.M[0][0] + M[1][1]*B.M[0][1] + M[2][1]*B.M[0][2];
			R.M[1][1] = M[0][1]*B.M[1][0] + M[1][1]*B.M[1][1] + M[2][1]*B.M[1][2];
			R.M[2][1] = M[0][1]*B.M[2][0] + M[1][1]*B.M[2][1] + M[2][1]*B.M[2][2];

			R.M[0][2] = M[0][2]*B.M[0][0] + M[1][2]*B.M[0][1] + M[2][2]*B.M[0][2];
			R.M[1][2] = M[0][2]*B.M[1][0] + M[1][2]*B.M[1][1] + M[2][2]*B.M[1][2];
			R.M[2][2] = M[0][2]*B.M[2][0] + M[1][2]*B.M[2][1] + M[2][2]*B.M[2][2];
			return R;
		}

		V2<T> Multiply(V2<T> &B)
		{
			V2<T> R;

			R.x = M[0][0]*B.x + M[1][0]*B.y + M[2][0];
			R.y = M[0][1]*B.x + M[1][1]*B.y + M[2][1];

			return R;
		};
	};
};

#endif

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
#ifndef O3_T_MAP_H
#define O3_T_MAP_H

namespace o3 {

template<typename K, typename V>
class tMap {
    struct Entry {
        K key;
        V val;

        Entry(const K& key, const V& val) : key(key), val(val)
        {
        }
    };

    struct Node {
        Node* parent;
        Node* left;
        Node* right;
        int height;
        Entry x;

        Node(Node* parent, const Entry& x) : parent(parent), left(0), right(0),
                                             height(1), x(x)
        {
        }

        int balance() const
        {
            return (right ? right->height : 0)
                 - (left  ? left->height  : 0);
        }

        void updateHeight()
        {
            height = max(left  ? left->height  : 0,
                         right ? right->height : 0) + 1;
        }

        void rotateLeft()
        {
            Node* pivot = right;
            Node* left = pivot->left;

            right = left;
            if (left)
                left->parent = this;
            if (parent) {
                if (parent->left == this)
                    parent->left = pivot;
                else
                    parent->right = pivot;
            }
            pivot->parent = parent;
            pivot->left  = this;
            parent = pivot;
            updateHeight();
            pivot->updateHeight();
        }

        void rotateRight()
        {
            Node* pivot = left;
            Node* right = pivot->right;

            left = right;
            if (right)
                right->parent = this;
            if (parent) {
                if (parent->left == this)
                    parent->left = pivot;
                else
                    parent->right = pivot;
            }
            pivot->parent = parent;
            pivot->right = this;
            parent = pivot;
            updateHeight();
            pivot->updateHeight();
        }

        void rebalance()                                                                 
        {
            int balance;
            Node* parent;

            if (!this->parent)
                return;
            updateHeight();
            balance = this->balance();
            if (balance < -1) {
                if (left->balance() > 0)
                    left->rotateLeft();
                rotateRight();
                parent = this->parent->parent;
            } else if (balance > 1) {
                if (right->balance() < 0)
                    right->rotateRight();
                rotateLeft();
                parent = this->parent->parent;
            } else
                parent = this->parent;
            if (parent)
                parent->rebalance();
        }
    };

    struct Impl {
        unsigned ref_count;
        size_t size;
        Node* head;
        Node* tail;

        Impl() : ref_count(1), size(0), head((Node*) /*malloc*/ memAlloc(sizeof(Node))),
                 tail(head)
        {
            head->parent = 0;
            head->left = 0;
            head->right = 0;
        }
    };

    Impl* m_impl;

    Node* findImpl(const K& key) const
    {
        Node* node = m_impl->tail;

        while (node) {
            if (node == m_impl->tail || key < node->x.key)
                node = node->left;
            else if (node->x.key < key)
                node = node->right;
            else
                return node;
        }
        return m_impl->tail;
    }
	
    void makeUnique()
    {
        if (m_impl->ref_count > 1) {
            tMap tmp;
			
            tmp.append(((const tMap*) this)->begin(),
					   ((const tMap*) this)->end());
            o3::swap(*this, tmp);
        }
    }

    Node* insertImpl(Node* parent, const Entry& x)
    {
        Node* new_node = o3_new(Node)(parent, x);

        ++m_impl->size;
        if (parent == m_impl->tail || x.key < parent->x.key) {
            parent->left = new_node;
            if (parent == m_impl->head)
                m_impl->head = new_node;
        } else
            parent->right = new_node;
        parent->rebalance();
        return new_node;
    }

public:
    class ConstIter {
        typename tMap::Node* m_node;

    public:
        ConstIter(typename tMap::Node* node = 0) : m_node(node)
        {
            o3_trace1 trace;
        }

        bool operator==(const ConstIter& that) const
        {
            o3_trace1 trace;

            return m_node == that.m_node;
        }

        bool operator!=(const ConstIter& that) const
        {
            o3_trace1 trace;

            return !(m_node == that.m_node);
        }

        bool valid() const
        {
            o3_trace1 trace;
 
            return m_node;
        }

        const typename tMap::Entry& operator*() const
        {
            o3_trace1 trace;

            return m_node->x;
        }

        const typename tMap::Entry* operator->() const
        {
            o3_trace1 trace;

            return &m_node->x;
        }

        ConstIter& operator++()
        {
            o3_trace1 trace;

            if (m_node->right) {
                m_node = m_node->right;
                while (m_node->left)
                    m_node = m_node->left;
            } else {
                Node* prev;

                do {
                    prev = m_node;
                    m_node = m_node->parent;
                } while (prev == m_node->right);
            }
            return *this;
        }

        ConstIter operator++(int)
        {
            o3_trace1 trace;
            ConstIter tmp = *this;

            ++*this;
            return tmp;
        }

        ConstIter& operator--()
        {
            o3_trace1 trace;

            if (m_node->left) {
                m_node = m_node->left;
                while (m_node->right)
                    m_node = m_node->right;
            } else {
                Node* prev;

                do {
                    prev = m_node;
                    m_node = m_node->parent;
                } while (prev == m_node->left);
            }
            return *this;
        }

        ConstIter operator--(int)
        {
            o3_trace1 trace;
            ConstIter tmp = *this;
    
            --*this;
            return tmp;
        }
     };

     class Iter {
        friend class tMap;

        typename tMap::Node* m_node;

    public:
        Iter(typename tMap::Node* node = 0) : m_node(node)
        {
            o3_trace1 trace;
        }

        bool operator==(const Iter& that) const
        {
            o3_trace1 trace;

            return m_node == that.m_node;
        }

        bool operator!=(const Iter& that) const
        {
            o3_trace1 trace;

            return !(*this == that);
        }

        bool valid() const
        {
            o3_trace1 trace;
 
            return m_node;
        }

        typename tMap::Entry& operator*() const
        {
            o3_trace1 trace;

            return m_node->x;
        }

        typename tMap::Entry* operator->() const
        {
            o3_trace1 trace;

            return &m_node->x;
        }

        operator ConstIter() const
        {
            o3_trace1 trace;

            return m_node;
        }

        Iter& operator++()
        {
            o3_trace1 trace;

            if (m_node->right) {
                m_node = m_node->right;
                while (m_node->left)
                    m_node = m_node->left;
            } else {
                Node* prev;

                do {
                    prev = m_node;
                    m_node = m_node->parent;
                } while (prev == m_node->right);
            }
            return *this;
        }

        Iter operator++(int)
        {
            o3_trace1 trace;
            Iter tmp = *this;

            ++*this;
            return tmp;
        }

        Iter& operator--()
        {
            o3_trace1 trace;

            if (m_node->left) {
                m_node = m_node->left;
                while (m_node->right)
                    m_node = m_node->right;
            } else {
                Node* prev;

                do {
                    prev = m_node;
                    m_node = m_node->parent;
                } while (prev == m_node->left);
            }
            return *this;
        }

        Iter operator--(int)
        {
            o3_trace trace;
            Iter tmp = *this;

            --*this;
            return tmp;
        }
    };

    tMap() : m_impl(o3_new(Impl)())
    {
        o3_trace1 trace;
    }

    tMap(const tMap& that) : m_impl(that.m_impl)
    {
        o3_trace1 trace;

        ++m_impl->ref_count;
    }

    tMap& operator=(const tMap& that)
    {
        o3_trace1 trace;
        tMap tmp = that;

        swap(tmp, *this);
        return *this;
    }

    ~tMap()
    {
        o3_trace1 trace;

        if (--m_impl->ref_count == 0) {
            clear();
            o3_delete(m_impl);
        }
    }

    bool empty() const
    {
        o3_trace1 trace;

        return size() == 0;
    }

    size_t size() const
    {
        o3_trace1 trace;

        return m_impl->size;
    }

    ConstIter begin() const
    {
        o3_trace1 trace;

        return m_impl->head;
    }

    ConstIter end() const
    {
        o3_trace1 trace;

        return m_impl->tail;
    }

    ConstIter find(const K& k) const
    {
        o3_trace1 trace;

        return findImpl(k);
    }

    Iter begin()
    {
        o3_trace1 trace;

		makeUnique();
        return m_impl->head;
    }

    Iter end()
    {
        o3_trace1 trace;

		makeUnique();
        return m_impl->tail;
    }

    Iter find(const K& k)
    {
        o3_trace1 trace;

        return findImpl(k);
    }

    Iter insert(const Entry& x)
    {
        o3_trace1 trace;
        Node* node;
        Node* parent  = 0;
        bool lt = false;
        Iter iter;

        for (node = m_impl->tail; node;
             node = lt ? node->left : node->right) {
            parent = node;
            lt = node == m_impl->tail || x.key < node->x.key;
        }
        iter = parent;
        if (lt) {
            if (parent == m_impl->head)
                return insertImpl(parent, x);
            else
                --iter;
        }
        if (iter->key < x.key)
            return insertImpl(parent, x);
        else
            return iter;
    }

    V& operator[](const K& k)
    {
        o3_trace1 trace;

        return insert(Entry(k, V()))->val;
    }

    Iter insert(Iter pos, const Entry& x)
    {
        o3_trace1 trace;
        Node* node = pos.m_node;
        Iter iter;

        if (node == m_impl->head) {
            if (x.key < node->x.key)
                return insert(node, x);
            else
                return insert(x);
        } else if (node == m_impl->tail) {
            if (node->x.key < x.key)
                return insert(node, x);
            else
                return insert(x);
        }
        iter = pos;
        --iter;
        if (iter->key < x.key && x.key < pos->key) {
            if (iter.m_node->right)
                return insertImpl(pos.m_node, x);
            else
                return insertImpl(iter.m_node, x);
        } else
            return insert(x);
    }

    void insert(Iter pos, const Entry& x, size_t n)
    {
        o3_trace1 trace;

        while (n--)
            pos = insert(pos, x);
    }

    void insert(Iter pos, ConstIter first, ConstIter last)
    {
        o3_trace1 trace;

        while (last != first)
            pos = insert(pos, *--last);
    }

	void append(const Entry& x, size_t n = 1)
    {
        o3_trace1 trace;
		
        insert(end(), x, n);
    }
	
    void append(ConstIter first, ConstIter last)
    {
        o3_trace1 trace;
		
        insert(end(), first, last);
    }
	
    bool remove(const K& key)
    {
        o3_trace1 trace;
        Iter iter = find(key);

        if (iter == end())
            return false;
        remove(iter);
        return true;
    }

    Iter remove(Iter pos)
    {
        o3_trace1 trace;
        Node* node;
        Node* parent;
        Node* child;

        node = pos.m_node;
        if (node->left && node->right) {
            node->x = *--pos;
            node = pos.m_node;
        }
        parent = node->parent;
        child = node->left ? node->left : node->right;
        ++pos;
        if (node == m_impl->head)
            m_impl->head = pos.m_node;
        o3_delete(node);
        if (parent) {
            if (node == parent->left)
                parent->left = child;
            else
                parent->right = child;
            parent->rebalance();
        }
        if (child)
            child->parent = parent;
        --m_impl->size;
        return pos;
    }

    void remove(Iter first, Iter last)
    {
        o3_trace1 trace;

        while (first != last)
            first = remove(first);
    }

    void clear()
    {
        o3_trace1 trace;

        remove(begin(), end());
    }
};

}

#endif // O3_T_MAP_H

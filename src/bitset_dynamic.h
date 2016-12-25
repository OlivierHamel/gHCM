
#pragma once


class BitsetDynamic {
    static size_t const kBitsPerWord = sizeof(size_t) * CHAR_BIT;
    static_assert(kBitsPerWord <= UCHAR_MAX, "BitIndex needs a bigger bitOffset...");

    struct BitIndex {
        size_t  wordIndex;
        uint8_t bitOffset;

        size_t mask() const { return size_t(1) << size_t(bitOffset); }
    };

    std::vector<size_t> _data;
    size_t              _length = 0;

    void dbg_validate() const {
        assert(_length <= (_data.size() * kBitsPerWord));
    }

    BitIndex bitIndex(size_t const bitIdx) const {
        dbg_validate();
        assert(bitIdx < size());
        return {         bitIdx / kBitsPerWord
               , uint8_t(bitIdx % kBitsPerWord) };
    }

    size_t bitIndexRev(BitIndex const& bi) const {
        dbg_validate();
        assert(bi.wordIndex < _data.size());
        assert(bi.bitOffset < kBitsPerWord);
        auto const bitIdx = bi.wordIndex * kBitsPerWord + bi.bitOffset;
        assert(bitIdx < size());
        return bitIdx;
    }

    size_t bitIndexNextSet(size_t const startIdx) const {
        assert((startIdx < size()) && "attempting to increment past end");
        if (startIdx >= size()) return startIdx; // already at end

        // return value undefined if word == 0
        /// \todo replace with a proper routine.
        auto const lsbSet = [](size_t const word) -> uint8_t {
            if (!word) return 0;

            for (uint8_t i = 1;; ++i) {
                auto const mask = size_t(1) << size_t(i);
                if (word & mask) return i;
            }

            assert(false && "unreachable");
            return 0;
        };

        auto const currInfo             = bitIndex(startIdx);
        auto const maskLessEqual        = (currInfo.mask() << size_t(1)) - size_t(1);
        auto const maskRemaining        = ~maskLessEqual;
        auto const currWord             = _data[currInfo.wordIndex];
        auto const currWordRemaining    = currWord & maskRemaining;
        if (currWordRemaining)
            return bitIndexRev({currInfo.wordIndex, lsbSet(currWordRemaining)});

        for (auto i = currInfo.wordIndex + 1; i < _data.size(); ++i) {
            auto const word = _data[i];
            if (word) return bitIndexRev({i, lsbSet(word)});
        }

        return size(); // we've reached the end Iter
    }

    template<typename Owner, typename TDeref>
    class Iter {
        friend class BitsetDynamic;

        Owner*  _owner       = nullptr;
        size_t  _bitIndex    = 0;

        Iter(Owner& owner, size_t offset) : _owner   (&owner )
                                          , _bitIndex(offset)
        { dbg_validate(); }

        void dbg_validate() const {
            assert(_owner || (_bitIndex == 0));
            if (_owner) {
                _owner->dbg_validate();
                assert(_bitIndex <= _owner->size()); // <=  since we need an end Iter
            }
        }

    public:
        Iter() = default;
        Iter(Iter const&) = default;

        Iter& operator=(Iter const& rhs) {
            assert(rhs._owner && "bitset dynamic iter should only be assigned initialised values");
            dbg_validate();
            rhs.dbg_validate();

            _owner      = rhs._owner;
            _bitIndex   = rhs._bitIndex;
            return *this;
        }

        TDeref    operator*()        { assert(_owner); return (*_owner)[_bitIndex]; }
        TDeref    operator*()  const { assert(_owner); return (*_owner)[_bitIndex]; }
        TDeref    operator->()       { assert(_owner); return (*_owner)[_bitIndex]; }
        TDeref    operator->() const { assert(_owner); return (*_owner)[_bitIndex]; }

        bool      operator!=(Iter const& rhs) const { return !(*this == rhs); }
        bool      operator==(Iter const& rhs) const {
            dbg_validate();
            rhs.dbg_validate();
            return (_owner      == rhs._owner   ) &&
                   (_bitIndex   == rhs._bitIndex);
        }

        Iter  operator++(int) { auto cpy = *this; ++(*this); return cpy; } // postfix
        Iter& operator++()    {
            dbg_validate();
            assert(_owner && "attempted to increment a uninitialised Iter");
            assert((_bitIndex < _owner->size()) && "attempting to increment past end");

            _bitIndex = _owner->bitIndexNextSet(_bitIndex);
            return *this;
        }
    };

public:
    class reference {
        friend class BitsetDynamic;
        BitsetDynamic& _owner;
        size_t         _bitIndex;

        reference(BitsetDynamic& owner, size_t offset) : _owner   (owner )
                                                       , _bitIndex(offset)
        { dbg_validate(); }

        void dbg_validate() const {
            _owner.dbg_validate();
            assert(_bitIndex < _owner.size());
        }

    public:
        reference() = delete;
        reference(reference const& m) : _owner   (m._owner )
                                      , _bitIndex(m._bitIndex)
        {}

        reference& operator=(reference const&) = delete;
        reference& operator=(bool const b) {
            dbg_validate();
            auto const bitIndex = _owner.bitIndex(_bitIndex);
            if (b) _owner._data[bitIndex.wordIndex] |=  bitIndex.mask();
            else   _owner._data[bitIndex.wordIndex] &= ~bitIndex.mask();
            return *this;
        }

        explicit operator bool() const {
            dbg_validate();
            auto const bitIndex = _owner.bitIndex(_bitIndex);
            return (_owner._data[bitIndex.wordIndex] & bitIndex.mask()) > 0;
        }

        void   flip()        { *this = !*this; }
        size_t index() const { dbg_validate(); return _bitIndex; }
    };

    using iterator          = Iter<BitsetDynamic        , reference >;
    using const_iterator    = Iter<BitsetDynamic const  , bool      >;


    BitsetDynamic() = default;
    BitsetDynamic(BitsetDynamic const&) = default;
    BitsetDynamic(BitsetDynamic&& m) : _data  (std::move(m._data    ))
                                     , _length(std::move(m._length  ))
    { dbg_validate(); }

    explicit BitsetDynamic(size_t const k) { resize(k); }

    BitsetDynamic(size_t const k, bool const initialValue) {
        resize(k);

        if (initialValue) {
            for (auto&& v : _data)
                v = std::numeric_limits<size_t>::max();
        }
    }

    BitsetDynamic& operator=(BitsetDynamic m) {
        std::swap(_data     , m._data  );
        std::swap(_length   , m._length);
        dbg_validate();
        return *this;
    }

    reference operator[](size_t const i)       { return { *this, i }; }
    bool      operator[](size_t const i) const
    { return bool(reference { const_cast<BitsetDynamic&>(*this), i }); }

    void resize(size_t const k) {
        size_t const whole   =  k / kBitsPerWord;
        size_t const partial = (k % kBitsPerWord) ? 1 : 0;

        _data.resize(whole + partial);
        _length = k;

        dbg_validate();
    }
    void    clear()      { std::fill(_data.begin(), _data.end(), 0); }
    size_t  size() const { return _length; }
    bool    any()  const { return begin() != end(); }

    iterator end()   { return { *this, size() }; }
    iterator begin() {
        if (size() == 0) return end();
        return { *this, (*this)[0] ? 0 : bitIndexNextSet(0) };
    }

    const_iterator end()    const { return { *this, size() }; }
    const_iterator begin()  const {
        if (size() == 0) return end();
        return { *this, (*this)[0] ? 0 : bitIndexNextSet(0) };
    }
    const_iterator cend()   const { return end();   }
    const_iterator cbegin() const { return begin(); }
};
/*****************************************************************************
 *  ___  _    _   ___ ___
 * |  _|| |  | | | _ \ _ \   CLIPP - command line interfaces for modern C++
 * | |_ | |_ | | |  _/  _/   version 1.2.0
 * |___||___||_| |_| |_|     https://github.com/muellan/clipp
 *
 * Licensed under the MIT License <http://opensource.org/licenses/MIT>.
 * Copyright (c) 2017-18 André Müller <foss@andremueller-online.de>
 *
 * ---------------------------------------------------------------------------
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef AM_CLIPP_H__
#define AM_CLIPP_H__

#include <cstring>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <memory>
#include <vector>
#include <limits>
#include <stack>
#include <algorithm>
#include <sstream>
#include <utility>
#include <iterator>
#include <functional>


/*************************************************************************//**
 *
 * @brief primary namespace
 *
 *****************************************************************************/
namespace clipp {



/*****************************************************************************
 *
 * basic constants and datatype definitions
 *
 *****************************************************************************/
using arg_index = int;

using arg_string = std::string;
using doc_string = std::string;

using arg_list  = std::vector<arg_string>;



/*************************************************************************//**
 *
 * @brief tristate
 *
 *****************************************************************************/
enum class tri : char { no, yes, either };

inline constexpr bool operator == (tri t, bool b) noexcept {
    return b ? t != tri::no : t != tri::yes;
}
inline constexpr bool operator == (bool b, tri t) noexcept { return  (t == b); }
inline constexpr bool operator != (tri t, bool b) noexcept { return !(t == b); }
inline constexpr bool operator != (bool b, tri t) noexcept { return !(t == b); }



/*************************************************************************//**
 *
 * @brief (start,size) index range
 *
 *****************************************************************************/
class subrange {
public:
    using size_type = arg_string::size_type;

    /** @brief default: no match */
    explicit constexpr
    subrange() noexcept :
        at_{arg_string::npos}, length_{0}
    {}

    /** @brief match length & position within subject string */
    explicit constexpr
    subrange(size_type pos, size_type len) noexcept :
        at_{pos}, length_{len}
    {}

    /** @brief position of the match within the subject string */
    constexpr size_type at()     const noexcept { return at_; }
    /** @brief length of the matching subsequence */
    constexpr size_type length() const noexcept { return length_; }

    /** @brief returns true, if query string is a prefix of the subject string */
    constexpr bool prefix() const noexcept {
        return at_ == 0 && length_ > 0;
    }

    /** @brief returns true, if query is a substring of the query string */
    constexpr explicit operator bool () const noexcept {
        return at_ != arg_string::npos && length_ > 0;
    }

private:
    size_type at_;
    size_type length_;
};



/*************************************************************************//**
 *
 * @brief match predicates
 *
 *****************************************************************************/
using match_predicate = std::function<bool(const arg_string&)>;
using match_function  = std::function<subrange(const arg_string&)>;






/*************************************************************************//**
 *
 * @brief type traits (NOT FOR DIRECT USE IN CLIENT CODE!)
 *        no interface guarantees; might be changed or removed in the future
 *
 *****************************************************************************/
namespace traits {

/*************************************************************************//**
 *
 * @brief function (class) signature type trait
 *
 *****************************************************************************/
template<class Fn, class Ret, class... Args>
constexpr auto
check_is_callable(int) -> decltype(
    std::declval<Fn>()(std::declval<Args>()...),
    std::integral_constant<bool,
        std::is_same<Ret,typename std::result_of<Fn(Args...)>::type>::value>{} );

template<class,class,class...>
constexpr auto
check_is_callable(long) -> std::false_type;

template<class Fn, class Ret>
constexpr auto
check_is_callable_without_arg(int) -> decltype(
    std::declval<Fn>()(),
    std::integral_constant<bool,
        std::is_same<Ret,typename std::result_of<Fn()>::type>::value>{} );

template<class,class>
constexpr auto
check_is_callable_without_arg(long) -> std::false_type;



template<class Fn, class... Args>
constexpr auto
check_is_void_callable(int) -> decltype(
    std::declval<Fn>()(std::declval<Args>()...), std::true_type{});

template<class,class,class...>
constexpr auto
check_is_void_callable(long) -> std::false_type;

template<class Fn>
constexpr auto
check_is_void_callable_without_arg(int) -> decltype(
    std::declval<Fn>()(), std::true_type{});

template<class>
constexpr auto
check_is_void_callable_without_arg(long) -> std::false_type;



template<class Fn, class Ret>
struct is_callable;


template<class Fn, class Ret, class... Args>
struct is_callable<Fn, Ret(Args...)> :
    decltype(check_is_callable<Fn,Ret,Args...>(0))
{};

template<class Fn, class Ret>
struct is_callable<Fn,Ret()> :
    decltype(check_is_callable_without_arg<Fn,Ret>(0))
{};


template<class Fn, class... Args>
struct is_callable<Fn, void(Args...)> :
    decltype(check_is_void_callable<Fn,Args...>(0))
{};

template<class Fn>
struct is_callable<Fn,void()> :
    decltype(check_is_void_callable_without_arg<Fn>(0))
{};



/*************************************************************************//**
 *
 * @brief input range type trait
 *
 *****************************************************************************/
template<class T>
constexpr auto
check_is_input_range(int) -> decltype(
    begin(std::declval<T>()), end(std::declval<T>()),
    std::true_type{});

template<class T>
constexpr auto
check_is_input_range(char) -> decltype(
    std::begin(std::declval<T>()), std::end(std::declval<T>()),
    std::true_type{});

template<class>
constexpr auto
check_is_input_range(long) -> std::false_type;

template<class T>
struct is_input_range :
    decltype(check_is_input_range<T>(0))
{};



/*************************************************************************//**
 *
 * @brief size() member type trait
 *
 *****************************************************************************/
template<class T>
constexpr auto
check_has_size_getter(int) ->
    decltype(std::declval<T>().size(), std::true_type{});

template<class>
constexpr auto
check_has_size_getter(long) -> std::false_type;

template<class T>
struct has_size_getter :
    decltype(check_has_size_getter<T>(0))
{};

} // namespace traits






/*************************************************************************//**
 *
 * @brief helpers (NOT FOR DIRECT USE IN CLIENT CODE!)
 *        no interface guarantees; might be changed or removed in the future
 *
 *****************************************************************************/
namespace detail {


/*************************************************************************//**
 * @brief forwards string to first non-whitespace char;
 *        std string -> unsigned conv yields max value, but we want 0;
 *        also checks for nullptr
 *****************************************************************************/
inline bool
fwd_to_unsigned_int(const char*& s)
{
    if(!s) return false;
    for(; std::isspace(*s); ++s);
    if(!s[0] || s[0] == '-') return false;
    if(s[0] == '-') return false;
    return true;
}


/*************************************************************************//**
 *
 * @brief value limits clamping
 *
 *****************************************************************************/
template<class T, class V, bool = (sizeof(V) > sizeof(T))>
struct limits_clamped {
    static T from(const V& v) {
        if(v >= V(std::numeric_limits<T>::max())) {
            return std::numeric_limits<T>::max();
        }
        if(v <= V(std::numeric_limits<T>::lowest())) {
            return std::numeric_limits<T>::lowest();
        }
        return T(v);
    }
};

template<class T, class V>
struct limits_clamped<T,V,false> {
    static T from(const V& v) { return T(v); }
};


/*************************************************************************//**
 *
 * @brief returns value of v as a T, clamped at T's maximum
 *
 *****************************************************************************/
template<class T, class V>
inline T clamped_on_limits(const V& v) {
    return limits_clamped<T,V>::from(v);
}




/*************************************************************************//**
 *
 * @brief type conversion helpers
 *
 *****************************************************************************/
template<class T>
struct make {
    static inline T from(const char* s) {
        if(!s) return false;
        //a conversion from const char* to / must exist
        return static_cast<T>(s);
    }
};

template<>
struct make<bool> {
    static inline bool from(const char* s) {
        if(!s) return false;
        return static_cast<bool>(s);
    }
};

template<>
struct make<unsigned char> {
    static inline unsigned char from(const char* s) {
        if(!fwd_to_unsigned_int(s)) return (0);
        return clamped_on_limits<unsigned char>(std::strtoull(s,nullptr,10));
    }
};

template<>
struct make<unsigned short int> {
    static inline unsigned short int from(const char* s) {
        if(!fwd_to_unsigned_int(s)) return (0);
        return clamped_on_limits<unsigned short int>(std::strtoull(s,nullptr,10));
    }
};

template<>
struct make<unsigned int> {
    static inline unsigned int from(const char* s) {
        if(!fwd_to_unsigned_int(s)) return (0);
        return clamped_on_limits<unsigned int>(std::strtoull(s,nullptr,10));
    }
};

template<>
struct make<unsigned long int> {
    static inline unsigned long int from(const char* s) {
        if(!fwd_to_unsigned_int(s)) return (0);
        return clamped_on_limits<unsigned long int>(std::strtoull(s,nullptr,10));
    }
};

template<>
struct make<unsigned long long int> {
    static inline unsigned long long int from(const char* s) {
        if(!fwd_to_unsigned_int(s)) return (0);
        return clamped_on_limits<unsigned long long int>(std::strtoull(s,nullptr,10));
    }
};

template<>
struct make<char> {
    static inline char from(const char* s) {
        //parse as single character?
        const auto n = std::strlen(s);
        if(n == 1) return s[0];
        //parse as integer
        return clamped_on_limits<char>(std::strtoll(s,nullptr,10));
    }
};

template<>
struct make<short int> {
    static inline short int from(const char* s) {
        return clamped_on_limits<short int>(std::strtoll(s,nullptr,10));
    }
};

template<>
struct make<int> {
    static inline int from(const char* s) {
        return clamped_on_limits<int>(std::strtoll(s,nullptr,10));
    }
};

template<>
struct make<long int> {
    static inline long int from(const char* s) {
        return clamped_on_limits<long int>(std::strtoll(s,nullptr,10));
    }
};

template<>
struct make<long long int> {
    static inline long long int from(const char* s) {
        return (std::strtoll(s,nullptr,10));
    }
};

template<>
struct make<float> {
    static inline float from(const char* s) {
        return (std::strtof(s,nullptr));
    }
};

template<>
struct make<double> {
    static inline double from(const char* s) {
        return (std::strtod(s,nullptr));
    }
};

template<>
struct make<long double> {
    static inline long double from(const char* s) {
        return (std::strtold(s,nullptr));
    }
};

template<>
struct make<std::string> {
    static inline std::string from(const char* s) {
        return std::string(s);
    }
};



/*************************************************************************//**
 *
 * @brief assigns boolean constant to one or multiple target objects
 *
 *****************************************************************************/
template<class T, class V = T>
class assign_value
{
public:
    template<class X>
    explicit constexpr
    assign_value(T& target, X&& value) noexcept :
        t_{std::addressof(target)}, v_{std::forward<X>(value)}
    {}

    void operator () () const {
        if(t_) *t_ = v_;
    }

private:
    T* t_;
    V v_;
};



/*************************************************************************//**
 *
 * @brief flips bools
 *
 *****************************************************************************/
class flip_bool
{
public:
    explicit constexpr
    flip_bool(bool& target) noexcept :
        b_{&target}
    {}

    void operator () () const {
        if(b_) *b_ = !*b_;
    }

private:
    bool* b_;
};



/*************************************************************************//**
 *
 * @brief increments using operator ++
 *
 *****************************************************************************/
template<class T>
class increment
{
public:
    explicit constexpr
    increment(T& target) noexcept : t_{std::addressof(target)} {}

    void operator () () const {
        if(t_) ++(*t_);
    }

private:
    T* t_;
};



/*************************************************************************//**
 *
 * @brief decrements using operator --
 *
 *****************************************************************************/
template<class T>
class decrement
{
public:
    explicit constexpr
    decrement(T& target) noexcept : t_{std::addressof(target)} {}

    void operator () () const {
        if(t_) --(*t_);
    }

private:
    T* t_;
};



/*************************************************************************//**
 *
 * @brief increments by a fixed amount using operator +=
 *
 *****************************************************************************/
template<class T>
class increment_by
{
public:
    explicit constexpr
    increment_by(T& target, T by) noexcept :
        t_{std::addressof(target)}, by_{std::move(by)}
    {}

    void operator () () const {
        if(t_) (*t_) += by_;
    }

private:
    T* t_;
    T by_;
};




/*************************************************************************//**
 *
 * @brief makes a value from a string and assigns it to an object
 *
 *****************************************************************************/
template<class T>
class map_arg_to
{
public:
    explicit constexpr
    map_arg_to(T& target) noexcept : t_{std::addressof(target)} {}

    void operator () (const char* s) const {
        if(t_ && s && (std::strlen(s) > 0))
            *t_ = detail::make<T>::from(s);
    }

private:
    T* t_;
};


//-------------------------------------------------------------------
/**
 * @brief specialization for vectors: append element
 */
template<class T>
class map_arg_to<std::vector<T>>
{
public:
    map_arg_to(std::vector<T>& target): t_{std::addressof(target)} {}

    void operator () (const char* s) const {
        if(t_ && s) t_->push_back(detail::make<T>::from(s));
    }

private:
    std::vector<T>* t_;
};


//-------------------------------------------------------------------
/**
 * @brief specialization for bools:
 *        set to true regardless of string content
 */
template<>
class map_arg_to<bool>
{
public:
    map_arg_to(bool& target): t_{&target} {}

    void operator () (const char* s) const {
        if(t_ && s) *t_ = true;
    }

private:
    bool* t_;
};


} // namespace detail






/*************************************************************************//**
 *
 * @brief string matching and processing tools
 *
 *****************************************************************************/

namespace str {


/*************************************************************************//**
 *
 * @brief converts string to value of target type 'T'
 *
 *****************************************************************************/
template<class T>
T make(const arg_string& s)
{
    return detail::make<T>::from(s);
}



/*************************************************************************//**
 *
 * @brief removes trailing whitespace from string
 *
 *****************************************************************************/
template<class C, class T, class A>
inline void
trimr(std::basic_string<C,T,A>& s)
{
    if(s.empty()) return;

    s.erase(
        std::find_if_not(s.rbegin(), s.rend(),
                         [](char c) { return std::isspace(c);} ).base(),
        s.end() );
}


/*************************************************************************//**
 *
 * @brief removes leading whitespace from string
 *
 *****************************************************************************/
template<class C, class T, class A>
inline void
triml(std::basic_string<C,T,A>& s)
{
    if(s.empty()) return;

    s.erase(
        s.begin(),
        std::find_if_not(s.begin(), s.end(),
                         [](char c) { return std::isspace(c);})
    );
}


/*************************************************************************//**
 *
 * @brief removes leading and trailing whitespace from string
 *
 *****************************************************************************/
template<class C, class T, class A>
inline void
trim(std::basic_string<C,T,A>& s)
{
    triml(s);
    trimr(s);
}


/*************************************************************************//**
 *
 * @brief removes all whitespaces from string
 *
 *****************************************************************************/
template<class C, class T, class A>
inline void
remove_ws(std::basic_string<C,T,A>& s)
{
    if(s.empty()) return;

    s.erase(std::remove_if(s.begin(), s.end(),
                           [](char c) { return std::isspace(c); }),
            s.end() );
}


/*************************************************************************//**
 *
 * @brief returns true, if the 'prefix' argument
 *        is a prefix of the 'subject' argument
 *
 *****************************************************************************/
template<class C, class T, class A>
inline bool
has_prefix(const std::basic_string<C,T,A>& subject,
           const std::basic_string<C,T,A>& prefix)
{
    if(prefix.size() > subject.size()) return false;
    return subject.find(prefix) == 0;
}


/*************************************************************************//**
 *
 * @brief returns true, if the 'postfix' argument
 *        is a postfix of the 'subject' argument
 *
 *****************************************************************************/
template<class C, class T, class A>
inline bool
has_postfix(const std::basic_string<C,T,A>& subject,
            const std::basic_string<C,T,A>& postfix)
{
    if(postfix.size() > subject.size()) return false;
    return (subject.size() - postfix.size()) == subject.find(postfix);
}



/*************************************************************************//**
*
* @brief   returns longest common prefix of several
*          sequential random access containers
*
* @details InputRange require begin and end (member functions or overloads)
*          the elements of InputRange require a size() member
*
*****************************************************************************/
template<class InputRange>
auto
longest_common_prefix(const InputRange& strs)
    -> typename std::decay<decltype(*begin(strs))>::type
{
    static_assert(traits::is_input_range<InputRange>(),
        "parameter must satisfy the InputRange concept");

    static_assert(traits::has_size_getter<
        typename std::decay<decltype(*begin(strs))>::type>(),
        "elements of input range must have a ::size() member function");

    using std::begin;
    using std::end;

    using item_t = typename std::decay<decltype(*begin(strs))>::type;
    using str_size_t = typename std::decay<decltype(begin(strs)->size())>::type;

    const auto n = size_t(distance(begin(strs), end(strs)));
    if(n < 1) return item_t("");
    if(n == 1) return *begin(strs);

    //length of shortest string
    auto m = std::min_element(begin(strs), end(strs),
                [](const item_t& a, const item_t& b) {
                    return a.size() < b.size(); })->size();

    //check each character until we find a mismatch
    for(str_size_t i = 0; i < m; ++i) {
        for(str_size_t j = 1; j < n; ++j) {
            if(strs[j][i] != strs[j-1][i])
                return strs[0].substr(0, i);
        }
    }
    return strs[0].substr(0, m);
}



/*************************************************************************//**
 *
 * @brief  returns longest substring range that could be found in 'arg'
 *
 * @param  arg         string to be searched in
 * @param  substrings  range of candidate substrings
 *
 *****************************************************************************/
template<class C, class T, class A, class InputRange>
subrange
longest_substring_match(const std::basic_string<C,T,A>& arg,
                        const InputRange& substrings)
{
    using string_t = std::basic_string<C,T,A>;

    static_assert(traits::is_input_range<InputRange>(),
        "parameter must satisfy the InputRange concept");

    static_assert(std::is_same<string_t,
        typename std::decay<decltype(*begin(substrings))>::type>(),
        "substrings must have same type as 'arg'");

    auto i = string_t::npos;
    auto n = string_t::size_type(0);
    for(const auto& s : substrings) {
        auto j = arg.find(s);
        if(j != string_t::npos && s.size() > n) {
            i = j;
            n = s.size();
        }
    }
    return subrange{i,n};
}



/*************************************************************************//**
 *
 * @brief  returns longest prefix range that could be found in 'arg'
 *
 * @param  arg       string to be searched in
 * @param  prefixes  range of candidate prefix strings
 *
 *****************************************************************************/
template<class C, class T, class A, class InputRange>
subrange
longest_prefix_match(const std::basic_string<C,T,A>& arg,
                     const InputRange& prefixes)
{
    using string_t = std::basic_string<C,T,A>;
    using s_size_t = typename string_t::size_type;

    static_assert(traits::is_input_range<InputRange>(),
        "parameter must satisfy the InputRange concept");

    static_assert(std::is_same<string_t,
        typename std::decay<decltype(*begin(prefixes))>::type>(),
        "prefixes must have same type as 'arg'");

    auto i = string_t::npos;
    auto n = s_size_t(0);
    for(const auto& s : prefixes) {
        auto j = arg.find(s);
        if(j == 0 && s.size() > n) {
            i = 0;
            n = s.size();
        }
    }
    return subrange{i,n};
}



/*************************************************************************//**
 *
 * @brief returns the first occurrence of 'query' within 'subject'
 *
 *****************************************************************************/
template<class C, class T, class A>
inline subrange
substring_match(const std::basic_string<C,T,A>& subject,
                const std::basic_string<C,T,A>& query)
{
    if(subject.empty() || query.empty()) return subrange{};
    auto i = subject.find(query);
    if(i == std::basic_string<C,T,A>::npos) return subrange{};
    return subrange{i,query.size()};
}



/*************************************************************************//**
 *
 * @brief returns first substring match (pos,len) within the input string
 *        that represents a number
 *        (with at maximum one decimal point and digit separators)
 *
 *****************************************************************************/
template<class C, class T, class A>
subrange
first_number_match(std::basic_string<C,T,A> s,
                   C digitSeparator = C(','),
                   C decimalPoint = C('.'),
                   C exponential = C('e'))
{
    using string_t = std::basic_string<C,T,A>;

    str::trim(s);
    if(s.empty()) return subrange{};

    auto i = s.find_first_of("0123456789+-");
    if(i == string_t::npos) {
        i = s.find(decimalPoint);
        if(i == string_t::npos) return subrange{};
    }

    bool point = false;
    bool sep = false;
    auto exp = string_t::npos;
    auto j = i + 1;
    for(; j < s.size(); ++j) {
        if(s[j] == digitSeparator) {
            if(!sep) sep = true; else break;
        }
        else {
            sep = false;
            if(s[j] == decimalPoint) {
                //only one decimal point before exponent allowed
                if(!point && exp == string_t::npos) point = true; else break;
            }
            else if(std::tolower(s[j]) == std::tolower(exponential)) {
                //only one exponent separator allowed
                if(exp == string_t::npos) exp = j; else break;
            }
            else if(exp != string_t::npos && (exp+1) == j) {
                //only sign or digit after exponent separator
                if(s[j] != '+' && s[j] != '-' && !std::isdigit(s[j])) break;
            }
            else if(!std::isdigit(s[j])) {
                break;
            }
        }
    }

    //if length == 1 then must be a digit
    if(j-i == 1 && !std::isdigit(s[i])) return subrange{};

    return subrange{i,j-i};
}



/*************************************************************************//**
 *
 * @brief returns first substring match (pos,len)
 *        that represents an integer (with optional digit separators)
 *
 *****************************************************************************/
template<class C, class T, class A>
subrange
first_integer_match(std::basic_string<C,T,A> s,
                    C digitSeparator = C(','))
{
    using string_t = std::basic_string<C,T,A>;

    str::trim(s);
    if(s.empty()) return subrange{};

    auto i = s.find_first_of("0123456789+-");
    if(i == string_t::npos) return subrange{};

    bool sep = false;
    auto j = i + 1;
    for(; j < s.size(); ++j) {
        if(s[j] == digitSeparator) {
            if(!sep) sep = true; else break;
        }
        else {
            sep = false;
            if(!std::isdigit(s[j])) break;
        }
    }

    //if length == 1 then must be a digit
    if(j-i == 1 && !std::isdigit(s[i])) return subrange{};

    return subrange{i,j-i};
}



/*************************************************************************//**
 *
 * @brief returns true if candidate string represents a number
 *
 *****************************************************************************/
template<class C, class T, class A>
bool represents_number(const std::basic_string<C,T,A>& candidate,
                   C digitSeparator = C(','),
                   C decimalPoint = C('.'),
                   C exponential = C('e'))
{
    const auto match = str::first_number_match(candidate, digitSeparator,
                                               decimalPoint, exponential);

    return (match && match.length() == candidate.size());
}



/*************************************************************************//**
 *
 * @brief returns true if candidate string represents an integer
 *
 *****************************************************************************/
template<class C, class T, class A>
bool represents_integer(const std::basic_string<C,T,A>& candidate,
                        C digitSeparator = C(','))
{
    const auto match = str::first_integer_match(candidate, digitSeparator);
    return (match && match.length() == candidate.size());
}

} // namespace str






/*************************************************************************//**
 *
 * @brief makes function object with a const char* parameter
 *        that assigns a value to a ref-captured object
 *
 *****************************************************************************/
template<class T, class V>
inline detail::assign_value<T,V>
set(T& target, V value) {
    return detail::assign_value<T>{target, std::move(value)};
}



/*************************************************************************//**
 *
 * @brief makes parameter-less function object
 *        that assigns value(s) to a ref-captured object;
 *        value(s) are obtained by converting the const char* argument to
 *        the captured object types;
 *        bools are always set to true if the argument is not nullptr
 *
 *****************************************************************************/
template<class T>
inline detail::map_arg_to<T>
set(T& target) {
    return detail::map_arg_to<T>{target};
}



/*************************************************************************//**
 *
 * @brief makes function object that sets a bool to true
 *
 *****************************************************************************/
inline detail::assign_value<bool>
set(bool& target) {
    return detail::assign_value<bool>{target,true};
}

/*************************************************************************//**
 *
 * @brief makes function object that sets a bool to false
 *
 *****************************************************************************/
inline detail::assign_value<bool>
unset(bool& target) {
    return detail::assign_value<bool>{target,false};
}

/*************************************************************************//**
 *
 * @brief makes function object that flips the value of a ref-captured bool
 *
 *****************************************************************************/
inline detail::flip_bool
flip(bool& b) {
    return detail::flip_bool(b);
}





/*************************************************************************//**
 *
 * @brief makes function object that increments using operator ++
 *
 *****************************************************************************/
template<class T>
inline detail::increment<T>
increment(T& target) {
    return detail::increment<T>{target};
}

/*************************************************************************//**
 *
 * @brief makes function object that decrements using operator --
 *
 *****************************************************************************/
template<class T>
inline detail::increment_by<T>
increment(T& target, T by) {
    return detail::increment_by<T>{target, std::move(by)};
}

/*************************************************************************//**
 *
 * @brief makes function object that increments by a fixed amount using operator +=
 *
 *****************************************************************************/
template<class T>
inline detail::decrement<T>
decrement(T& target) {
    return detail::decrement<T>{target};
}






/*************************************************************************//**
 *
 * @brief helpers (NOT FOR DIRECT USE IN CLIENT CODE!)
 *
 *****************************************************************************/
namespace detail {


/*************************************************************************//**
 *
 * @brief mixin that provides action definition and execution
 *
 *****************************************************************************/
template<class Derived>
class action_provider
{
private:
    //---------------------------------------------------------------
    using simple_action = std::function<void()>;
    using arg_action    = std::function<void(const char*)>;
    using index_action  = std::function<void(int)>;

    //-----------------------------------------------------
    class simple_action_adapter {
    public:
        simple_action_adapter() = default;
        simple_action_adapter(const simple_action& a): action_(a) {}
        simple_action_adapter(simple_action&& a): action_(std::move(a)) {}
        void operator() (const char*) const { action_(); }
        void operator() (int) const { action_(); }
    private:
        simple_action action_;
    };


public:
    //---------------------------------------------------------------
    /** @brief adds an action that has an operator() that is callable
     *         with a 'const char*' argument */
    Derived&
    call(arg_action a) {
        argActions_.push_back(std::move(a));
        return *static_cast<Derived*>(this);
    }

    /** @brief adds an action that has an operator()() */
    Derived&
    call(simple_action a) {
        argActions_.push_back(simple_action_adapter(std::move(a)));
        return *static_cast<Derived*>(this);
    }

    /** @brief adds an action that has an operator() that is callable
     *         with a 'const char*' argument */
    Derived& operator () (arg_action a)    { return call(std::move(a)); }

    /** @brief adds an action that has an operator()() */
    Derived& operator () (simple_action a) { return call(std::move(a)); }


    //---------------------------------------------------------------
    /** @brief adds an action that will set the value of 't' from
     *         a 'const char*' arg */
    template<class Target>
    Derived&
    set(Target& t) {
        static_assert(!std::is_pointer<Target>::value,
                      "parameter target type must not be a pointer");

        return call(clipp::set(t));
    }

    /** @brief adds an action that will set the value of 't' to 'v' */
    template<class Target, class Value>
    Derived&
    set(Target& t, Value&& v) {
        return call(clipp::set(t, std::forward<Value>(v)));
    }


    //---------------------------------------------------------------
    /** @brief adds an action that will be called if a parameter
     *         matches an argument for the 2nd, 3rd, 4th, ... time
     */
    Derived&
    if_repeated(simple_action a) {
        repeatActions_.push_back(simple_action_adapter{std::move(a)});
        return *static_cast<Derived*>(this);
    }
    /** @brief adds an action that will be called with the argument's
     *         index if a parameter matches an argument for
     *         the 2nd, 3rd, 4th, ... time
     */
    Derived&
    if_repeated(index_action a) {
        repeatActions_.push_back(std::move(a));
        return *static_cast<Derived*>(this);
    }


    //---------------------------------------------------------------
    /** @brief adds an action that will be called if a required parameter
     *         is missing
     */
    Derived&
    if_missing(simple_action a) {
        missingActions_.push_back(simple_action_adapter{std::move(a)});
        return *static_cast<Derived*>(this);
    }
    /** @brief adds an action that will be called if a required parameter
     *         is missing; the action will get called with the index of
     *         the command line argument where the missing event occured first
     */
    Derived&
    if_missing(index_action a) {
        missingActions_.push_back(std::move(a));
        return *static_cast<Derived*>(this);
    }


    //---------------------------------------------------------------
    /** @brief adds an action that will be called if a parameter
     *         was matched, but was unreachable in the current scope
     */
    Derived&
    if_blocked(simple_action a) {
        blockedActions_.push_back(simple_action_adapter{std::move(a)});
        return *static_cast<Derived*>(this);
    }
    /** @brief adds an action that will be called if a parameter
     *         was matched, but was unreachable in the current scope;
     *         the action will be called with the index of
     *         the command line argument where the problem occured
     */
    Derived&
    if_blocked(index_action a) {
        blockedActions_.push_back(std::move(a));
        return *static_cast<Derived*>(this);
    }


    //---------------------------------------------------------------
    /** @brief adds an action that will be called if a parameter match
     *         was in conflict with a different alternative parameter
     */
    Derived&
    if_conflicted(simple_action a) {
        conflictActions_.push_back(simple_action_adapter{std::move(a)});
        return *static_cast<Derived*>(this);
    }
    /** @brief adds an action that will be called if a parameter match
     *         was in conflict with a different alternative paramete;
     *         the action will be called with the index of
     *         the command line argument where the problem occuredr
     */
    Derived&
    if_conflicted(index_action a) {
        conflictActions_.push_back(std::move(a));
        return *static_cast<Derived*>(this);
    }


    //---------------------------------------------------------------
    /** @brief adds targets = either objects whose values should be
     *         set by command line arguments or actions that should
     *         be called in case of a match */
    template<class T, class... Ts>
    Derived&
    target(T&& t, Ts&&... ts) {
        target(std::forward<T>(t));
        target(std::forward<Ts>(ts)...);
        return *static_cast<Derived*>(this);
    }

    /** @brief adds action that should be called in case of a match */
    template<class T, class = typename std::enable_if<
            !std::is_fundamental<typename std::decay<T>::type>() &&
            (traits::is_callable<T,void()>() ||
             traits::is_callable<T,void(const char*)>() )
        >::type>
    Derived&
    target(T&& t) {
        call(std::forward<T>(t));
        return *static_cast<Derived*>(this);
    }

    /** @brief adds object whose value should be set by command line arguments
     */
    template<class T, class = typename std::enable_if<
            std::is_fundamental<typename std::decay<T>::type>() ||
            (!traits::is_callable<T,void()>() &&
             !traits::is_callable<T,void(const char*)>() )
        >::type>
    Derived&
    target(T& t) {
        set(t);
        return *static_cast<Derived*>(this);
    }

    //TODO remove ugly empty param list overload
    Derived&
    target() {
        return *static_cast<Derived*>(this);
    }


    //---------------------------------------------------------------
    /** @brief adds target, see member function 'target' */
    template<class Target>
    inline friend Derived&
    operator << (Target&& t, Derived& p) {
        p.target(std::forward<Target>(t));
        return p;
    }
    /** @brief adds target, see member function 'target' */
    template<class Target>
    inline friend Derived&&
    operator << (Target&& t, Derived&& p) {
        p.target(std::forward<Target>(t));
        return std::move(p);
    }

    //-----------------------------------------------------
    /** @brief adds target, see member function 'target' */
    template<class Target>
    inline friend Derived&
    operator >> (Derived& p, Target&& t) {
        p.target(std::forward<Target>(t));
        return p;
    }
    /** @brief adds target, see member function 'target' */
    template<class Target>
    inline friend Derived&&
    operator >> (Derived&& p, Target&& t) {
        p.target(std::forward<Target>(t));
        return std::move(p);
    }


    //---------------------------------------------------------------
    /** @brief executes all argument actions */
    void execute_actions(const arg_string& arg) const {
        int i = 0;
        for(const auto& a : argActions_) {
            ++i;
            a(arg.c_str());
        }
    }

    /** @brief executes repeat actions */
    void notify_repeated(arg_index idx) const {
        for(const auto& a : repeatActions_) a(idx);
    }
    /** @brief executes missing error actions */
    void notify_missing(arg_index idx) const {
        for(const auto& a : missingActions_) a(idx);
    }
    /** @brief executes blocked error actions */
    void notify_blocked(arg_index idx) const {
        for(const auto& a : blockedActions_) a(idx);
    }
    /** @brief executes conflict error actions */
    void notify_conflict(arg_index idx) const {
        for(const auto& a : conflictActions_) a(idx);
    }

private:
    //---------------------------------------------------------------
    std::vector<arg_action> argActions_;
    std::vector<index_action> repeatActions_;
    std::vector<index_action> missingActions_;
    std::vector<index_action> blockedActions_;
    std::vector<index_action> conflictActions_;
};






/*************************************************************************//**
 *
 * @brief mixin that provides basic common settings of parameters and groups
 *
 *****************************************************************************/
template<class Derived>
class token
{
public:
    //---------------------------------------------------------------
    using doc_string = clipp::doc_string;


    //---------------------------------------------------------------
    /** @brief returns documentation string */
    const doc_string& doc() const noexcept {
        return doc_;
    }

    /** @brief sets documentations string */
    Derived& doc(const doc_string& txt) {
        doc_ = txt;
        return *static_cast<Derived*>(this);
    }

    /** @brief sets documentations string */
    Derived& doc(doc_string&& txt) {
        doc_ = std::move(txt);
        return *static_cast<Derived*>(this);
    }


    //---------------------------------------------------------------
    /** @brief returns if a group/parameter is repeatable */
    bool repeatable() const noexcept {
        return repeatable_;
    }

    /** @brief sets repeatability of group/parameter */
    Derived& repeatable(bool yes) noexcept {
        repeatable_ = yes;
        return *static_cast<Derived*>(this);
    }


    //---------------------------------------------------------------
    /** @brief returns if a group/parameter is blocking/positional */
    bool blocking() const noexcept {
        return blocking_;
    }

    /** @brief determines, if a group/parameter is blocking/positional */
    Derived& blocking(bool yes) noexcept {
        blocking_ = yes;
        return *static_cast<Derived*>(this);
    }


private:
    //---------------------------------------------------------------
    doc_string doc_;
    bool repeatable_ = false;
    bool blocking_ = false;
};




/*************************************************************************//**
 *
 * @brief sets documentation strings on a token
 *
 *****************************************************************************/
template<class T>
inline T&
operator % (doc_string docstr, token<T>& p)
{
    return p.doc(std::move(docstr));
}
//---------------------------------------------------------
template<class T>
inline T&&
operator % (doc_string docstr, token<T>&& p)
{
    return std::move(p.doc(std::move(docstr)));
}

//---------------------------------------------------------
template<class T>
inline T&
operator % (token<T>& p, doc_string docstr)
{
    return p.doc(std::move(docstr));
}
//---------------------------------------------------------
template<class T>
inline T&&
operator % (token<T>&& p, doc_string docstr)
{
    return std::move(p.doc(std::move(docstr)));
}




/*************************************************************************//**
 *
 * @brief sets documentation strings on a token
 *
 *****************************************************************************/
template<class T>
inline T&
doc(doc_string docstr, token<T>& p)
{
    return p.doc(std::move(docstr));
}
//---------------------------------------------------------
template<class T>
inline T&&
doc(doc_string docstr, token<T>&& p)
{
    return std::move(p.doc(std::move(docstr)));
}



} // namespace detail



/*************************************************************************//**
 *
 * @brief contains parameter matching functions and function classes
 *
 *****************************************************************************/
namespace match {


/*************************************************************************//**
 *
 * @brief predicate that is always true
 *
 *****************************************************************************/
inline bool
any(const arg_string&) { return true; }

/*************************************************************************//**
 *
 * @brief predicate that is always false
 *
 *****************************************************************************/
inline bool
none(const arg_string&) { return false; }



/*************************************************************************//**
 *
 * @brief predicate that returns true if the argument string is non-empty string
 *
 *****************************************************************************/
inline bool
nonempty(const arg_string& s) {
    return !s.empty();
}



/*************************************************************************//**
 *
 * @brief predicate that returns true if the argument is a non-empty
 *        string that consists only of alphanumeric characters
 *
 *****************************************************************************/
inline bool
alphanumeric(const arg_string& s) {
    if(s.empty()) return false;
    return std::all_of(s.begin(), s.end(), [](char c) {return std::isalnum(c); });
}



/*************************************************************************//**
 *
 * @brief predicate that returns true if the argument is a non-empty
 *        string that consists only of alphabetic characters
 *
 *****************************************************************************/
inline bool
alphabetic(const arg_string& s) {
    return std::all_of(s.begin(), s.end(), [](char c) {return std::isalpha(c); });
}



/*************************************************************************//**
 *
 * @brief predicate that returns false if the argument string is
 *        equal to any string from the exclusion list
 *
 *****************************************************************************/
class none_of
{
public:
    none_of(arg_list strs):
        excluded_{std::move(strs)}
    {}

    template<class... Strings>
    none_of(arg_string str, Strings&&... strs):
        excluded_{std::move(str), std::forward<Strings>(strs)...}
    {}

    template<class... Strings>
    none_of(const char* str, Strings&&... strs):
        excluded_{arg_string(str), std::forward<Strings>(strs)...}
    {}

    bool operator () (const arg_string& arg) const {
        return (std::find(begin(excluded_), end(excluded_), arg)
                == end(excluded_));
    }

private:
    arg_list excluded_;
};



/*************************************************************************//**
 *
 * @brief predicate that returns the first substring match within the input
 *        string that rmeepresents a number
 *        (with at maximum one decimal point and digit separators)
 *
 *****************************************************************************/
class numbers
{
public:
    explicit
    numbers(char decimalPoint = '.',
            char digitSeparator = ' ',
            char exponentSeparator = 'e')
    :
        decpoint_{decimalPoint}, separator_{digitSeparator},
        exp_{exponentSeparator}
    {}

    subrange operator () (const arg_string& s) const {
        return str::first_number_match(s, separator_, decpoint_, exp_);
    }

private:
    char decpoint_;
    char separator_;
    char exp_;
};



/*************************************************************************//**
 *
 * @brief predicate that returns true if the input string represents an integer
 *        (with optional digit separators)
 *
 *****************************************************************************/
class integers {
public:
    explicit
    integers(char digitSeparator = ' '): separator_{digitSeparator} {}

    subrange operator () (const arg_string& s) const {
        return str::first_integer_match(s, separator_);
    }

private:
    char separator_;
};



/*************************************************************************//**
 *
 * @brief predicate that returns true if the input string represents
 *        a non-negative integer (with optional digit separators)
 *
 *****************************************************************************/
class positive_integers {
public:
    explicit
    positive_integers(char digitSeparator = ' '): separator_{digitSeparator} {}

    subrange operator () (const arg_string& s) const {
        auto match = str::first_integer_match(s, separator_);
        if(!match) return subrange{};
        if(s[match.at()] == '-') return subrange{};
        return match;
    }

private:
    char separator_;
};



/*************************************************************************//**
 *
 * @brief predicate that returns true if the input string
 *        contains a given substring
 *
 *****************************************************************************/
class substring
{
public:
    explicit
    substring(arg_string str): str_{std::move(str)} {}

    subrange operator () (const arg_string& s) const {
        return str::substring_match(s, str_);
    }

private:
    arg_string str_;
};



/*************************************************************************//**
 *
 * @brief predicate that returns true if the input string starts
 *        with a given prefix
 *
 *****************************************************************************/
class prefix {
public:
    explicit
    prefix(arg_string p): prefix_{std::move(p)} {}

    bool operator () (const arg_string& s) const {
        return s.find(prefix_) == 0;
    }

private:
    arg_string prefix_;
};



/*************************************************************************//**
 *
 * @brief predicate that returns true if the input string does not start
 *        with a given prefix
 *
 *****************************************************************************/
class prefix_not {
public:
    explicit
    prefix_not(arg_string p): prefix_{std::move(p)} {}

    bool operator () (const arg_string& s) const {
        return s.find(prefix_) != 0;
    }

private:
    arg_string prefix_;
};


/** @brief alias for prefix_not */
using noprefix = prefix_not;



/*************************************************************************//**
 *
 * @brief predicate that returns true if the length of the input string
 *        is wihtin a given interval
 *
 *****************************************************************************/
class length {
public:
    explicit
    length(std::size_t exact):
        min_{exact}, max_{exact}
    {}

    explicit
    length(std::size_t min, std::size_t max):
        min_{min}, max_{max}
    {}

    bool operator () (const arg_string& s) const {
        return s.size() >= min_ && s.size() <= max_;
    }

private:
    std::size_t min_;
    std::size_t max_;
};


/*************************************************************************//**
 *
 * @brief makes function object that returns true if the input string has a
 *        given minimum length
 *
 *****************************************************************************/
inline length min_length(std::size_t min)
{
    return length{min, arg_string::npos-1};
}

/*************************************************************************//**
 *
 * @brief makes function object that returns true if the input string is
 *        not longer than a given maximum length
 *
 *****************************************************************************/
inline length max_length(std::size_t max)
{
    return length{0, max};
}


} // namespace match





/*************************************************************************//**
 *
 * @brief command line parameter that can match one or many arguments.
 *
 *****************************************************************************/
class parameter :
    public detail::token<parameter>,
    public detail::action_provider<parameter>
{
    /** @brief adapts a 'match_predicate' to the 'match_function' interface */
    class predicate_adapter {
    public:
        explicit
        predicate_adapter(match_predicate pred): match_{std::move(pred)} {}

        subrange operator () (const arg_string& arg) const {
            return match_(arg) ? subrange{0,arg.size()} : subrange{};
        }

    private:
        match_predicate match_;
    };

public:
    //---------------------------------------------------------------
    /** @brief makes default parameter, that will match nothing */
    parameter():
        flags_{},
        matcher_{predicate_adapter{match::none}},
        label_{}, required_{false}, greedy_{false}
    {}

    /** @brief makes "flag" parameter */
    template<class... Strings>
    explicit
    parameter(arg_string str, Strings&&... strs):
        flags_{},
        matcher_{predicate_adapter{match::none}},
        label_{}, required_{false}, greedy_{false}
    {
        add_flags(std::move(str), std::forward<Strings>(strs)...);
    }

    /** @brief makes "flag" parameter from range of strings */
    explicit
    parameter(const arg_list& flaglist):
        flags_{},
        matcher_{predicate_adapter{match::none}},
        label_{}, required_{false}, greedy_{false}
    {
        add_flags(flaglist);
    }

    //-----------------------------------------------------
    /** @brief makes "value" parameter with custom match predicate
     *         (= yes/no matcher)
     */
    explicit
    parameter(match_predicate filter):
        flags_{},
        matcher_{predicate_adapter{std::move(filter)}},
        label_{}, required_{false}, greedy_{false}
    {}

    /** @brief makes "value" parameter with custom match function
     *         (= partial matcher)
     */
    explicit
    parameter(match_function filter):
        flags_{},
        matcher_{std::move(filter)},
        label_{}, required_{false}, greedy_{false}
    {}


    //---------------------------------------------------------------
    /** @brief returns if a parameter is required */
    bool
    required() const noexcept {
        return required_;
    }

    /** @brief determines if a parameter is required */
    parameter&
    required(bool yes) noexcept {
        required_ = yes;
        return *this;
    }


    //---------------------------------------------------------------
    /** @brief returns if a parameter should match greedily */
    bool
    greedy() const noexcept {
        return greedy_;
    }

    /** @brief determines if a parameter should match greedily */
    parameter&
    greedy(bool yes) noexcept {
        greedy_ = yes;
        return *this;
    }


    //---------------------------------------------------------------
    /** @brief returns parameter label;
     *         will be used for documentation, if flags are empty
     */
    const doc_string&
    label() const {
        return label_;
    }

    /** @brief sets parameter label;
     *         will be used for documentation, if flags are empty
     */
    parameter&
    label(const doc_string& lbl) {
        label_ = lbl;
        return *this;
    }

    /** @brief sets parameter label;
     *         will be used for documentation, if flags are empty
     */
    parameter&
    label(doc_string&& lbl) {
        label_ = lbl;
        return *this;
    }


    //---------------------------------------------------------------
    /** @brief returns either longest matching prefix of 'arg' in any
     *         of the flags or the result of the custom match operation
     */
    subrange
    match(const arg_string& arg) const
    {
        if(arg.empty()) return subrange{};

        if(flags_.empty()) {
            return matcher_(arg);
        }
        else {
            if(std::find(flags_.begin(), flags_.end(), arg) != flags_.end()) {
                return subrange{0,arg.size()};
            }
            return str::longest_prefix_match(arg, flags_);
        }
    }


    //---------------------------------------------------------------
    /** @brief access range of flag strings */
    const arg_list&
    flags() const noexcept {
        return flags_;
    }

    /** @brief access custom match operation */
    const match_function&
    matcher() const noexcept {
        return matcher_;
    }


    //---------------------------------------------------------------
    /** @brief prepend prefix to each flag */
    inline friend parameter&
    with_prefix(const arg_string& prefix, parameter& p)
    {
        if(prefix.empty() || p.flags().empty()) return p;

        for(auto& f : p.flags_) {
            if(f.find(prefix) != 0) f.insert(0, prefix);
        }
        return p;
    }


    /** @brief prepend prefix to each flag
     */
    inline friend parameter&
    with_prefixes_short_long(
        const arg_string& shortpfx, const arg_string& longpfx,
        parameter& p)
    {
        if(shortpfx.empty() && longpfx.empty()) return p;
        if(p.flags().empty()) return p;

        for(auto& f : p.flags_) {
            if(f.size() == 1) {
                if(f.find(shortpfx) != 0) f.insert(0, shortpfx);
            } else {
                if(f.find(longpfx) != 0) f.insert(0, longpfx);
            }
        }
        return p;
    }

private:
    //---------------------------------------------------------------
    void add_flags(arg_string str) {
        //empty flags are not allowed
        str::remove_ws(str);
        if(!str.empty()) flags_.push_back(std::move(str));
    }

    //---------------------------------------------------------------
    void add_flags(const arg_list& strs) {
        if(strs.empty()) return;
        flags_.reserve(flags_.size() + strs.size());
        for(const auto& s : strs) add_flags(s);
    }

    template<class String1, class String2, class... Strings>
    void
    add_flags(String1&& s1, String2&& s2, Strings&&... ss) {
        flags_.reserve(2 + sizeof...(ss));
        add_flags(std::forward<String1>(s1));
        add_flags(std::forward<String2>(s2), std::forward<Strings>(ss)...);
    }

    arg_list flags_;
    match_function matcher_;
    doc_string label_;
    bool required_ = false;
    bool greedy_ = false;
};




/*************************************************************************//**
 *
 * @brief makes required non-blocking exact match parameter
 *
 *****************************************************************************/
template<class String, class... Strings>
inline parameter
command(String&& flag, Strings&&... flags)
{
    return parameter{std::forward<String>(flag), std::forward<Strings>(flags)...}
        .required(true).blocking(true).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes required non-blocking exact match parameter
 *
 *****************************************************************************/
template<class String, class... Strings>
inline parameter
required(String&& flag, Strings&&... flags)
{
    return parameter{std::forward<String>(flag), std::forward<Strings>(flags)...}
        .required(true).blocking(false).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes optional, non-blocking exact match parameter
 *
 *****************************************************************************/
template<class String, class... Strings>
inline parameter
option(String&& flag, Strings&&... flags)
{
    return parameter{std::forward<String>(flag), std::forward<Strings>(flags)...}
        .required(false).blocking(false).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes required, blocking, repeatable value parameter;
 *        matches any non-empty string
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
value(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::nonempty}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(false);
}

template<class Filter, class... Targets, class = typename std::enable_if<
    traits::is_callable<Filter,bool(const char*)>::value ||
    traits::is_callable<Filter,subrange(const char*)>::value>::type>
inline parameter
value(Filter&& filter, doc_string label, Targets&&... tgts)
{
    return parameter{std::forward<Filter>(filter)}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes required, blocking, repeatable value parameter;
 *        matches any non-empty string
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
values(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::nonempty}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(true);
}

template<class Filter, class... Targets, class = typename std::enable_if<
    traits::is_callable<Filter,bool(const char*)>::value ||
    traits::is_callable<Filter,subrange(const char*)>::value>::type>
inline parameter
values(Filter&& filter, doc_string label, Targets&&... tgts)
{
    return parameter{std::forward<Filter>(filter)}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes optional, blocking value parameter;
 *        matches any non-empty string
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
opt_value(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::nonempty}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(false);
}

template<class Filter, class... Targets, class = typename std::enable_if<
    traits::is_callable<Filter,bool(const char*)>::value ||
    traits::is_callable<Filter,subrange(const char*)>::value>::type>
inline parameter
opt_value(Filter&& filter, doc_string label, Targets&&... tgts)
{
    return parameter{std::forward<Filter>(filter)}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes optional, blocking, repeatable value parameter;
 *        matches any non-empty string
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
opt_values(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::nonempty}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(true);
}

template<class Filter, class... Targets, class = typename std::enable_if<
    traits::is_callable<Filter,bool(const char*)>::value ||
    traits::is_callable<Filter,subrange(const char*)>::value>::type>
inline parameter
opt_values(Filter&& filter, doc_string label, Targets&&... tgts)
{
    return parameter{std::forward<Filter>(filter)}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes required, blocking value parameter;
 *        matches any string consisting of alphanumeric characters
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
word(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::alphanumeric}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes required, blocking, repeatable value parameter;
 *        matches any string consisting of alphanumeric characters
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
words(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::alphanumeric}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes optional, blocking value parameter;
 *        matches any string consisting of alphanumeric characters
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
opt_word(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::alphanumeric}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes optional, blocking, repeatable value parameter;
 *        matches any string consisting of alphanumeric characters
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
opt_words(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::alphanumeric}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes required, blocking value parameter;
 *        matches any string that represents a number
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
number(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::numbers{}}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes required, blocking, repeatable value parameter;
 *        matches any string that represents a number
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
numbers(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::numbers{}}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes optional, blocking value parameter;
 *        matches any string that represents a number
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
opt_number(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::numbers{}}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes optional, blocking, repeatable value parameter;
 *        matches any string that represents a number
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
opt_numbers(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::numbers{}}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes required, blocking value parameter;
 *        matches any string that represents an integer
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
integer(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::integers{}}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes required, blocking, repeatable value parameter;
 *        matches any string that represents an integer
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
integers(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::integers{}}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(true).blocking(true).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes optional, blocking value parameter;
 *        matches any string that represents an integer
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
opt_integer(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::integers{}}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(false);
}



/*************************************************************************//**
 *
 * @brief makes optional, blocking, repeatable value parameter;
 *        matches any string that represents an integer
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
opt_integers(const doc_string& label, Targets&&... tgts)
{
    return parameter{match::integers{}}
        .label(label)
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes catch-all value parameter
 *
 *****************************************************************************/
template<class... Targets>
inline parameter
any_other(Targets&&... tgts)
{
    return parameter{match::any}
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes catch-all value parameter with custom filter
 *
 *****************************************************************************/
template<class Filter, class... Targets, class = typename std::enable_if<
    traits::is_callable<Filter,bool(const char*)>::value ||
    traits::is_callable<Filter,subrange(const char*)>::value>::type>
inline parameter
any(Filter&& filter, Targets&&... tgts)
{
    return parameter{std::forward<Filter>(filter)}
        .target(std::forward<Targets>(tgts)...)
        .required(false).blocking(false).repeatable(true);
}




/*************************************************************************//**
 *
 * @brief group of parameters and/or other groups;
 *        can be configured to act as a group of alternatives (exclusive match)
 *
 *****************************************************************************/
class group :
    public detail::token<group>
{
    //---------------------------------------------------------------
    /**
        * @brief tagged union type that either stores a parameter or a group
        *        and provides a common interface to them
        *        could be replaced by std::variant in the future
        *
        *        Note to future self: do NOT try again to do this with
        *        dynamic polymorphism; there are a couple of
        *        nasty problems associated with it and the implementation
        *        becomes bloated and needlessly complicated.
        */
    template<class Param, class Group>
    struct child_t {
        enum class type : char {param, group};
    public:

        explicit
        child_t(const Param&  v)          : m_{v},            type_{type::param} {}
        child_t(      Param&& v) noexcept : m_{std::move(v)}, type_{type::param} {}

        explicit
        child_t(const Group&  g)          : m_{g},            type_{type::group} {}
        child_t(      Group&& g) noexcept : m_{std::move(g)}, type_{type::group} {}

        child_t(const child_t& src): type_{src.type_} {
            switch(type_) {
                default:
                case type::param: new(&m_)data{src.m_.param}; break;
                case type::group: new(&m_)data{src.m_.group}; break;
            }
        }

        child_t(child_t&& src) noexcept : type_{src.type_} {
            switch(type_) {
                default:
                case type::param: new(&m_)data{std::move(src.m_.param)}; break;
                case type::group: new(&m_)data{std::move(src.m_.group)}; break;
            }
        }

        child_t& operator = (const child_t& src) {
            destroy_content();
            type_ = src.type_;
            switch(type_) {
                default:
                case type::param: new(&m_)data{src.m_.param}; break;
                case type::group: new(&m_)data{src.m_.group}; break;
            }
            return *this;
        }

        child_t& operator = (child_t&& src) noexcept {
            destroy_content();
            type_ = src.type_;
            switch(type_) {
                default:
                case type::param: new(&m_)data{std::move(src.m_.param)}; break;
                case type::group: new(&m_)data{std::move(src.m_.group)}; break;
            }
            return *this;
        }

        ~child_t() {
            destroy_content();
        }

        const doc_string&
        doc() const noexcept {
            switch(type_) {
                default:
                case type::param: return m_.param.doc();
                case type::group: return m_.group.doc();
            }
        }

        bool blocking() const noexcept {
            switch(type_) {
                case type::param: return m_.param.blocking();
                case type::group: return m_.group.blocking();
                default: return false;
            }
        }
        bool repeatable() const noexcept {
            switch(type_) {
                case type::param: return m_.param.repeatable();
                case type::group: return m_.group.repeatable();
                default: return false;
            }
        }
        bool required() const noexcept {
            switch(type_) {
                case type::param: return m_.param.required();
                case type::group:
                    return (m_.group.exclusive() && m_.group.all_required() ) ||
                          (!m_.group.exclusive() && m_.group.any_required()  );
                default: return false;
            }
        }
        bool exclusive() const noexcept {
            switch(type_) {
                case type::group: return m_.group.exclusive();
                case type::param:
                default: return false;
            }
        }
        std::size_t param_count() const noexcept {
            switch(type_) {
                case type::group: return m_.group.param_count();
                case type::param:
                default: return std::size_t(1);
            }
        }
        std::size_t depth() const noexcept {
            switch(type_) {
                case type::group: return m_.group.depth();
                case type::param:
                default: return std::size_t(0);
            }
        }

        void execute_actions(const arg_string& arg) const {
            switch(type_) {
                default:
                case type::group: return;
                case type::param: m_.param.execute_actions(arg); break;
            }

        }

        void notify_repeated(arg_index idx) const {
            switch(type_) {
                default:
                case type::group: return;
                case type::param: m_.param.notify_repeated(idx); break;
            }
        }
        void notify_missing(arg_index idx) const {
            switch(type_) {
                default:
                case type::group: return;
                case type::param: m_.param.notify_missing(idx); break;
            }
        }
        void notify_blocked(arg_index idx) const {
            switch(type_) {
                default:
                case type::group: return;
                case type::param: m_.param.notify_blocked(idx); break;
            }
        }
        void notify_conflict(arg_index idx) const {
            switch(type_) {
                default:
                case type::group: return;
                case type::param: m_.param.notify_conflict(idx); break;
            }
        }

        bool is_param() const noexcept { return type_ == type::param; }
        bool is_group() const noexcept { return type_ == type::group; }

        Param& as_param() noexcept { return m_.param; }
        Group& as_group() noexcept { return m_.group; }

        const Param& as_param() const noexcept { return m_.param; }
        const Group& as_group() const noexcept { return m_.group; }

    private:
        void destroy_content() {
            switch(type_) {
                default:
                case type::param: m_.param.~Param(); break;
                case type::group: m_.group.~Group(); break;
            }
        }

        union data {
            data() {}

            data(const Param&  v)          : param{v} {}
            data(      Param&& v) noexcept : param{std::move(v)} {}

            data(const Group&  g)          : group{g} {}
            data(      Group&& g) noexcept : group{std::move(g)} {}
            ~data() {}

            Param param;
            Group group;
        };

        data m_;
        type type_;
    };


public:
    //---------------------------------------------------------------
    using child      = child_t<parameter,group>;
    using value_type = child;

private:
    using children_store = std::vector<child>;

public:
    using const_iterator = children_store::const_iterator;
    using iterator       = children_store::iterator;
    using size_type      = children_store::size_type;


    //---------------------------------------------------------------
    /**
     * @brief recursively iterates over all nodes
     */
    class depth_first_traverser
    {
    public:
        //-----------------------------------------------------
        struct context {
            context() = default;
            context(const group& p):
                parent{&p}, cur{p.begin()}, end{p.end()}
            {}
            const group* parent = nullptr;
            const_iterator cur;
            const_iterator end;
        };
        using context_list = std::vector<context>;

        //-----------------------------------------------------
        class memento {
            friend class depth_first_traverser;
            int level_;
            context context_;
        public:
            int level() const noexcept { return level_; }
            const child* param() const noexcept { return &(*context_.cur); }
        };

        depth_first_traverser() = default;

        explicit
        depth_first_traverser(const group& cur): stack_{} {
            if(!cur.empty()) stack_.emplace_back(cur);
        }

        explicit operator bool() const noexcept {
            return !stack_.empty();
        }

        int level() const noexcept {
            return int(stack_.size());
        }

        bool is_first_in_group() const noexcept {
            if(stack_.empty()) return false;
            return (stack_.back().cur == stack_.back().parent->begin());
        }

        bool is_last_in_group() const noexcept {
            if(stack_.empty()) return false;
            return (stack_.back().cur+1 == stack_.back().end);
        }

        bool is_last_in_path() const noexcept {
            if(stack_.empty()) return false;
            for(const auto& t : stack_) {
                if(t.cur+1 != t.end) return false;
            }
            const auto& top = stack_.back();
            //if we have to descend into group on next ++ => not last in path
            if(top.cur->is_group()) return false;
            return true;
        }

        /** @brief inside a group of alternatives >= minlevel */
        bool is_alternative(int minlevel = 0) const noexcept {
            if(stack_.empty()) return false;
            if(minlevel > 0) minlevel -= 1;
            if(minlevel >= int(stack_.size())) return false;
            return std::any_of(stack_.begin() + minlevel, stack_.end(),
                [](const context& c) { return c.parent->exclusive(); });
        }

        /** @brief repeatable or inside a repeatable group >= minlevel */
        bool is_repeatable(int minlevel = 0) const noexcept {
            if(stack_.empty()) return false;
            if(stack_.back().cur->repeatable()) return true;
            if(minlevel > 0) minlevel -= 1;
            if(minlevel >= int(stack_.size())) return false;
            return std::any_of(stack_.begin() + minlevel, stack_.end(),
                [](const context& c) { return c.parent->repeatable(); });
        }
        /** @brief inside group with joinable flags */
        bool joinable() const noexcept {
            if(stack_.empty()) return false;
            return std::any_of(stack_.begin(), stack_.end(),
                [](const context& c) { return c.parent->joinable(); });
        }

        const context_list&
        stack() const {
            return stack_;
        }

        /** @brief innermost repeat group */
        const group*
        repeat_group() const noexcept {
            auto i = std::find_if(stack_.rbegin(), stack_.rend(),
                [](const context& c) { return c.parent->repeatable(); });

            return i != stack_.rend() ? i->parent : nullptr;
        }

        /** @brief outermost join group */
        const group*
        join_group() const noexcept {
            auto i = std::find_if(stack_.begin(), stack_.end(),
                [](const context& c) { return c.parent->joinable(); });
            return i != stack_.end() ? i->parent : nullptr;
        }

        const group* root() const noexcept {
            return stack_.empty() ? nullptr : stack_.front().parent;
        }

        /** @brief common flag prefix of all flags in current group */
        arg_string common_flag_prefix() const noexcept {
            if(stack_.empty()) return "";
            auto g = join_group();
            return g ? g->common_flag_prefix() : arg_string("");
        }

        const child&
        operator * () const noexcept {
            return *stack_.back().cur;
        }

        const child*
        operator -> () const noexcept {
            return &(*stack_.back().cur);
        }

        const group&
        parent() const noexcept {
            return *(stack_.back().parent);
        }


        /** @brief go to next element of depth first search */
        depth_first_traverser&
        operator ++ () {
            if(stack_.empty()) return *this;
            //at group -> decend into group
            if(stack_.back().cur->is_group()) {
                stack_.emplace_back(stack_.back().cur->as_group());
            }
            else {
                next_sibling();
            }
            return *this;
        }

        /** @brief go to next sibling of current */
        depth_first_traverser&
        next_sibling() {
            if(stack_.empty()) return *this;
            ++stack_.back().cur;
            //at the end of current group?
            while(stack_.back().cur == stack_.back().end) {
                //go to parent
                stack_.pop_back();
                if(stack_.empty()) return *this;
                //go to next sibling in parent
                ++stack_.back().cur;
            }
            return *this;
        }

        /** @brief go to next position after siblings of current */
        depth_first_traverser&
        next_after_siblings() {
            if(stack_.empty()) return *this;
            stack_.back().cur = stack_.back().end-1;
            next_sibling();
            return *this;
        }

        /** @brief skips to next alternative in innermost group
        */
        depth_first_traverser&
        next_alternative() {
            if(stack_.empty()) return *this;

            //find first exclusive group (from the top of the stack!)
            auto i = std::find_if(stack_.rbegin(), stack_.rend(),
                [](const context& c) { return c.parent->exclusive(); });
            if(i == stack_.rend()) return *this;

            stack_.erase(i.base(), stack_.end());
            next_sibling();
            return *this;
        }

        /**
         * @brief
         */
        depth_first_traverser&
        back_to_parent() {
            if(stack_.empty()) return *this;
            stack_.pop_back();
            return *this;
        }

        /** @brief don't visit next siblings, go back to parent on next ++
         *         note: renders siblings unreachable for *this
         **/
        depth_first_traverser&
        skip_siblings() {
            if(stack_.empty()) return *this;
            //future increments won't visit subsequent siblings:
            stack_.back().end = stack_.back().cur+1;
            return *this;
        }

        /** @brief skips all other alternatives in surrounding exclusive groups
         *         on next ++
         *         note: renders alternatives unreachable for *this
        */
        depth_first_traverser&
        skip_alternatives() {
            if(stack_.empty()) return *this;

            //exclude all other alternatives in surrounding groups
            //by making their current position the last one
            for(auto& c : stack_) {
                if(c.parent && c.parent->exclusive() && c.cur < c.end)
                    c.end = c.cur+1;
            }

            return *this;
        }

        void invalidate() {
            stack_.clear();
        }

        inline friend bool operator == (const depth_first_traverser& a,
                                        const depth_first_traverser& b)
        {
            if(a.stack_.empty() || b.stack_.empty()) return false;

            //parents not the same -> different position
            if(a.stack_.back().parent != b.stack_.back().parent) return false;

            bool aEnd = a.stack_.back().cur == a.stack_.back().end;
            bool bEnd = b.stack_.back().cur == b.stack_.back().end;
            //either both at the end of the same parent => same position
            if(aEnd && bEnd) return true;
            //or only one at the end => not at the same position
            if(aEnd || bEnd) return false;
            return std::addressof(*a.stack_.back().cur) ==
                   std::addressof(*b.stack_.back().cur);
        }
        inline friend bool operator != (const depth_first_traverser& a,
                                        const depth_first_traverser& b)
        {
            return !(a == b);
        }

        memento
        undo_point() const {
            memento m;
            m.level_ = int(stack_.size());
            if(!stack_.empty()) m.context_ = stack_.back();
            return m;
        }

        void undo(const memento& m) {
            if(m.level_ < 1) return;
            if(m.level_ <= int(stack_.size())) {
                stack_.erase(stack_.begin() + m.level_, stack_.end());
                stack_.back() = m.context_;
            }
            else if(stack_.empty() && m.level_ == 1) {
                stack_.push_back(m.context_);
            }
        }

    private:
        context_list stack_;
    };


    //---------------------------------------------------------------
    group() = default;

    template<class Param, class... Params>
    explicit
    group(doc_string docstr, Param param, Params... params):
        children_{}, exclusive_{false}, joinable_{false}, scoped_{true}
    {
        doc(std::move(docstr));
        push_back(std::move(param), std::move(params)...);
    }

    template<class... Params>
    explicit
    group(parameter param, Params... params):
        children_{}, exclusive_{false}, joinable_{false}, scoped_{true}
    {
        push_back(std::move(param), std::move(params)...);
    }

    template<class P2, class... Ps>
    explicit
    group(group p1, P2 p2, Ps... ps):
        children_{}, exclusive_{false}, joinable_{false}, scoped_{true}
    {
        push_back(std::move(p1), std::move(p2), std::move(ps)...);
    }


    //-----------------------------------------------------
    group(const group&) = default;
    group(group&&) = default;


    //---------------------------------------------------------------
    group& operator = (const group&) = default;
    group& operator = (group&&) = default;


    //---------------------------------------------------------------
    /** @brief determines if a command line argument can be matched by a
     *         combination of (partial) matches through any number of children
     */
    group& joinable(bool yes) {
        joinable_ = yes;
        return *this;
    }

    /** @brief returns if a command line argument can be matched by a
     *         combination of (partial) matches through any number of children
     */
    bool joinable() const noexcept {
        return joinable_;
    }


    //---------------------------------------------------------------
    /** @brief turns explicit scoping on or off
     *         operators , & | and other combinating functions will
     *         not merge groups that are marked as scoped
     */
    group& scoped(bool yes) {
        scoped_ = yes;
        return *this;
    }

    /** @brief returns true if operators , & | and other combinating functions
     *         will merge groups and false otherwise
     */
    bool scoped() const noexcept
    {
        return scoped_;
    }


    //---------------------------------------------------------------
    /** @brief determines if children are mutually exclusive alternatives */
    group& exclusive(bool yes) {
        exclusive_ = yes;
        return *this;
    }
    /** @brief returns if children are mutually exclusive alternatives */
    bool exclusive() const noexcept {
        return exclusive_;
    }


    //---------------------------------------------------------------
    /** @brief returns true, if any child is required to match */
    bool any_required() const
    {
        return std::any_of(children_.begin(), children_.end(),
            [](const child& n){ return n.required(); });
    }
    /** @brief returns true, if all children are required to match */
    bool all_required() const
    {
        return std::all_of(children_.begin(), children_.end(),
            [](const child& n){ return n.required(); });
    }


    //---------------------------------------------------------------
    /** @brief returns true if any child is optional (=non-required) */
    bool any_optional() const {
        return !all_required();
    }
    /** @brief returns true if all children are optional (=non-required) */
    bool all_optional() const {
        return !any_required();
    }


    //---------------------------------------------------------------
    /** @brief returns if the entire group is blocking / positional */
    bool blocking() const noexcept {
        return token<group>::blocking() || (exclusive() && all_blocking());
    }
    //-----------------------------------------------------
    /** @brief determines if the entire group is blocking / positional */
    group& blocking(bool yes) {
        return token<group>::blocking(yes);
    }

    //---------------------------------------------------------------
    /** @brief returns true if any child is blocking */
    bool any_blocking() const
    {
        return std::any_of(children_.begin(), children_.end(),
            [](const child& n){ return n.blocking(); });
    }
    //---------------------------------------------------------------
    /** @brief returns true if all children is blocking */
    bool all_blocking() const
    {
        return std::all_of(children_.begin(), children_.end(),
            [](const child& n){ return n.blocking(); });
    }


    //---------------------------------------------------------------
    /** @brief returns if any child is a value parameter (recursive) */
    bool any_flagless() const
    {
        return std::any_of(children_.begin(), children_.end(),
            [](const child& p){
                return p.is_param() && p.as_param().flags().empty();
            });
    }
    /** @brief returns if all children are value parameters (recursive) */
    bool all_flagless() const
    {
        return std::all_of(children_.begin(), children_.end(),
            [](const child& p){
                return p.is_param() && p.as_param().flags().empty();
            });
    }


    //---------------------------------------------------------------
    /** @brief adds child parameter at the end */
    group&
    push_back(const parameter& v) {
        children_.emplace_back(v);
        return *this;
    }
    //-----------------------------------------------------
    /** @brief adds child parameter at the end */
    group&
    push_back(parameter&& v) {
        children_.emplace_back(std::move(v));
        return *this;
    }
    //-----------------------------------------------------
    /** @brief adds child group at the end */
    group&
    push_back(const group& g) {
        children_.emplace_back(g);
        return *this;
    }
    //-----------------------------------------------------
    /** @brief adds child group at the end */
    group&
    push_back(group&& g) {
        children_.emplace_back(std::move(g));
        return *this;
    }


    //-----------------------------------------------------
    /** @brief adds children (groups and/or parameters) */
    template<class Param1, class Param2, class... Params>
    group&
    push_back(Param1&& param1, Param2&& param2, Params&&... params)
    {
        children_.reserve(children_.size() + 2 + sizeof...(params));
        push_back(std::forward<Param1>(param1));
        push_back(std::forward<Param2>(param2), std::forward<Params>(params)...);
        return *this;
    }


    //---------------------------------------------------------------
    /** @brief adds child parameter at the beginning */
    group&
    push_front(const parameter& v) {
        children_.emplace(children_.begin(), v);
        return *this;
    }
    //-----------------------------------------------------
    /** @brief adds child parameter at the beginning */
    group&
    push_front(parameter&& v) {
        children_.emplace(children_.begin(), std::move(v));
        return *this;
    }
    //-----------------------------------------------------
    /** @brief adds child group at the beginning */
    group&
    push_front(const group& g) {
        children_.emplace(children_.begin(), g);
        return *this;
    }
    //-----------------------------------------------------
    /** @brief adds child group at the beginning */
    group&
    push_front(group&& g) {
        children_.emplace(children_.begin(), std::move(g));
        return *this;
    }


    //---------------------------------------------------------------
    /** @brief adds all children of other group at the end */
    group&
    merge(group&& g)
    {
        children_.insert(children_.end(),
                      std::make_move_iterator(g.begin()),
                      std::make_move_iterator(g.end()));
        return *this;
    }
    //-----------------------------------------------------
    /** @brief adds all children of several other groups at the end */
    template<class... Groups>
    group&
    merge(group&& g1, group&& g2, Groups&&... gs)
    {
        merge(std::move(g1));
        merge(std::move(g2), std::forward<Groups>(gs)...);
        return *this;
    }


    //---------------------------------------------------------------
    /** @brief indexed, nutable access to child */
    child& operator [] (size_type index) noexcept {
        return children_[index];
    }
    /** @brief indexed, non-nutable access to child */
    const child& operator [] (size_type index) const noexcept {
        return children_[index];
    }

    //---------------------------------------------------------------
    /** @brief mutable access to first child */
          child& front()       noexcept { return children_.front(); }
    /** @brief non-mutable access to first child */
    const child& front() const noexcept { return children_.front(); }
    //-----------------------------------------------------
    /** @brief mutable access to last child */
          child& back()       noexcept { return children_.back(); }
    /** @brief non-mutable access to last child */
    const child& back() const noexcept { return children_.back(); }


    //---------------------------------------------------------------
    /** @brief returns true, if group has no children, false otherwise */
    bool empty() const noexcept     { return children_.empty(); }

    /** @brief returns number of children */
    size_type size() const noexcept { return children_.size(); }

    /** @brief returns number of nested levels; 1 for a flat group */
    size_type depth() const {
        size_type n = 0;
        for(const auto& c : children_) {
            auto l = 1 + c.depth();
            if(l > n) n = l;
        }
        return n;
    }


    //---------------------------------------------------------------
    /** @brief returns mutating iterator to position of first element */
          iterator  begin()       noexcept { return children_.begin(); }
    /** @brief returns non-mutating iterator to position of first element */
    const_iterator  begin() const noexcept { return children_.begin(); }
    /** @brief returns non-mutating iterator to position of first element */
    const_iterator cbegin() const noexcept { return children_.begin(); }

    /** @brief returns mutating iterator to position one past the last element */
          iterator  end()         noexcept { return children_.end(); }
    /** @brief returns non-mutating iterator to position one past the last element */
    const_iterator  end()   const noexcept { return children_.end(); }
    /** @brief returns non-mutating iterator to position one past the last element */
    const_iterator cend()   const noexcept { return children_.end(); }


    //---------------------------------------------------------------
    /** @brief returns augmented iterator for depth first searches
     *  @details taverser knows end of iteration and can skip over children
     */
    depth_first_traverser
    begin_dfs() const noexcept {
        return depth_first_traverser{*this};
    }


    //---------------------------------------------------------------
    /** @brief returns recursive parameter count */
    size_type param_count() const {
        size_type c = 0;
        for(const auto& n : children_) {
            c += n.param_count();
        }
        return c;
    }


    //---------------------------------------------------------------
    /** @brief returns range of all flags (recursive) */
    arg_list all_flags() const
    {
        std::vector<arg_string> all;
        gather_flags(children_, all);
        return all;
    }

    /** @brief returns true, if no flag occurs as true
     *         prefix of any other flag (identical flags will be ignored) */
    bool flags_are_prefix_free() const
    {
        const auto fs = all_flags();

        using std::begin; using std::end;
        for(auto i = begin(fs), e = end(fs); i != e; ++i) {
            if(!i->empty()) {
                for(auto j = i+1; j != e; ++j) {
                    if(!j->empty() && *i != *j) {
                        if(i->find(*j) == 0) return false;
                        if(j->find(*i) == 0) return false;
                    }
                }
            }
        }

        return true;
    }


    //---------------------------------------------------------------
    /** @brief returns longest common prefix of all flags */
    arg_string common_flag_prefix() const
    {
        arg_list prefixes;
        gather_prefixes(children_, prefixes);
        return str::longest_common_prefix(prefixes);
    }


private:
    //---------------------------------------------------------------
    static void
    gather_flags(const children_store& nodes, arg_list& all)
    {
        for(const auto& p : nodes) {
            if(p.is_group()) {
                gather_flags(p.as_group().children_, all);
            }
            else {
                const auto& pf = p.as_param().flags();
                using std::begin;
                using std::end;
                if(!pf.empty()) all.insert(end(all), begin(pf), end(pf));
            }
        }
    }
    //---------------------------------------------------------------
    static void
    gather_prefixes(const children_store& nodes, arg_list& all)
    {
        for(const auto& p : nodes) {
            if(p.is_group()) {
                gather_prefixes(p.as_group().children_, all);
            }
            else if(!p.as_param().flags().empty()) {
                auto pfx = str::longest_common_prefix(p.as_param().flags());
                if(!pfx.empty()) all.push_back(std::move(pfx));
            }
        }
    }

    //---------------------------------------------------------------
    children_store children_;
    bool exclusive_ = false;
    bool joinable_ = false;
    bool scoped_ = false;
};



/*************************************************************************//**
 *
 * @brief group or parameter
 *
 *****************************************************************************/
using pattern = group::child;



/*************************************************************************//**
 *
 * @brief apply an action to all parameters in a group
 *
 *****************************************************************************/
template<class Action>
void for_all_params(group& g, Action&& action)
{
    for(auto& p : g) {
        if(p.is_group()) {
            for_all_params(p.as_group(), action);
        }
        else {
            action(p.as_param());
        }
    }
}

template<class Action>
void for_all_params(const group& g, Action&& action)
{
    for(auto& p : g) {
        if(p.is_group()) {
            for_all_params(p.as_group(), action);
        }
        else {
            action(p.as_param());
        }
    }
}



/*************************************************************************//**
 *
 * @brief makes a group of parameters and/or groups
 *
 *****************************************************************************/
inline group
operator , (parameter a, parameter b)
{
    return group{std::move(a), std::move(b)}.scoped(false);
}

//---------------------------------------------------------
inline group
operator , (parameter a, group b)
{
    return !b.scoped() && !b.blocking() && !b.exclusive() && !b.repeatable()
        && !b.joinable() && (b.doc().empty() || b.doc() == a.doc())
       ? b.push_front(std::move(a))
       : group{std::move(a), std::move(b)}.scoped(false);
}

//---------------------------------------------------------
inline group
operator , (group a, parameter b)
{
    return !a.scoped() && !a.blocking() && !a.exclusive() && !a.repeatable()
        && !a.joinable() && (a.doc().empty() || a.doc() == b.doc())
       ? a.push_back(std::move(b))
       : group{std::move(a), std::move(b)}.scoped(false);
}

//---------------------------------------------------------
inline group
operator , (group a, group b)
{
    return !a.scoped() && !a.blocking() && !a.exclusive() && !a.repeatable()
        && !a.joinable() && (a.doc().empty() || a.doc() == b.doc())
       ? a.push_back(std::move(b))
       : group{std::move(a), std::move(b)}.scoped(false);
}



/*************************************************************************//**
 *
 * @brief makes a group of alternative parameters or groups
 *
 *****************************************************************************/
template<class Param, class... Params>
inline group
one_of(Param param, Params... params)
{
    return group{std::move(param), std::move(params)...}.exclusive(true);
}


/*************************************************************************//**
 *
 * @brief makes a group of alternative parameters or groups
 *
 *****************************************************************************/
inline group
operator | (parameter a, parameter b)
{
    return group{std::move(a), std::move(b)}.scoped(false).exclusive(true);
}

//-------------------------------------------------------------------
inline group
operator | (parameter a, group b)
{
    return !b.scoped() && !b.blocking() && b.exclusive() && !b.repeatable()
        && !b.joinable()
        && (b.doc().empty() || b.doc() == a.doc())
        ? b.push_front(std::move(a))
        : group{std::move(a), std::move(b)}.scoped(false).exclusive(true);
}

//-------------------------------------------------------------------
inline group
operator | (group a, parameter b)
{
    return !a.scoped() && a.exclusive() && !a.repeatable() && !a.joinable()
        && a.blocking() == b.blocking()
        && (a.doc().empty() || a.doc() == b.doc())
        ? a.push_back(std::move(b))
        : group{std::move(a), std::move(b)}.scoped(false).exclusive(true);
}

inline group
operator | (group a, group b)
{
    return !a.scoped() && a.exclusive() &&!a.repeatable() && !a.joinable()
        && a.blocking() == b.blocking()
        && (a.doc().empty() || a.doc() == b.doc())
        ? a.push_back(std::move(b))
        : group{std::move(a), std::move(b)}.scoped(false).exclusive(true);
}



/*************************************************************************//**
 *
 * @brief helpers (NOT FOR DIRECT USE IN CLIENT CODE!)
 *        no interface guarantees; might be changed or removed in the future
 *
 *****************************************************************************/
namespace detail {

inline void set_blocking(bool) {}

template<class P, class... Ps>
void set_blocking(bool yes, P& p, Ps&... ps) {
    p.blocking(yes);
    set_blocking(yes, ps...);
}

} // namespace detail


/*************************************************************************//**
 *
 * @brief makes a parameter/group sequence by making all input objects blocking
 *
 *****************************************************************************/
template<class Param, class... Params>
inline group
in_sequence(Param param, Params... params)
{
    detail::set_blocking(true, param, params...);
    return group{std::move(param), std::move(params)...}.scoped(true);
}


/*************************************************************************//**
 *
 * @brief makes a parameter/group sequence by making all input objects blocking
 *
 *****************************************************************************/
inline group
operator & (parameter a, parameter b)
{
    a.blocking(true);
    b.blocking(true);
    return group{std::move(a), std::move(b)}.scoped(true);
}

//---------------------------------------------------------
inline group
operator & (parameter a, group b)
{
    a.blocking(true);
    return group{std::move(a), std::move(b)}.scoped(true);
}

//---------------------------------------------------------
inline group
operator & (group a, parameter b)
{
    b.blocking(true);
    if(a.all_blocking() && !a.exclusive() && !a.repeatable() && !a.joinable()
        && (a.doc().empty() || a.doc() == b.doc()))
    {
        return a.push_back(std::move(b));
    }
    else {
        if(!a.all_blocking()) a.blocking(true);
        return group{std::move(a), std::move(b)}.scoped(true);
    }
}

inline group
operator & (group a, group b)
{
    if(!b.all_blocking()) b.blocking(true);
    if(a.all_blocking() && !a.exclusive() && !a.repeatable()
        && !a.joinable() && (a.doc().empty() || a.doc() == b.doc()))
    {
        return a.push_back(std::move(b));
    }
    else {
        if(!a.all_blocking()) a.blocking(true);
        return group{std::move(a), std::move(b)}.scoped(true);
    }
}



/*************************************************************************//**
 *
 * @brief makes a group of parameters and/or groups
 *        where all single char flag params ("-a", "b", ...) are joinable
 *
 *****************************************************************************/
inline group
joinable(group g) {
    return g.joinable(true);
}

//-------------------------------------------------------------------
template<class... Params>
inline group
joinable(parameter param, Params... params)
{
    return group{std::move(param), std::move(params)...}.joinable(true);
}

template<class P2, class... Ps>
inline group
joinable(group p1, P2 p2, Ps... ps)
{
    return group{std::move(p1), std::move(p2), std::move(ps)...}.joinable(true);
}

template<class Param, class... Params>
inline group
joinable(doc_string docstr, Param param, Params... params)
{
    return group{std::move(param), std::move(params)...}
                .joinable(true).doc(std::move(docstr));
}



/*************************************************************************//**
 *
 * @brief makes a repeatable copy of a parameter
 *
 *****************************************************************************/
inline parameter
repeatable(parameter p) {
    return p.repeatable(true);
}

/*************************************************************************//**
 *
 * @brief makes a repeatable copy of a group
 *
 *****************************************************************************/
inline group
repeatable(group g) {
    return g.repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes a group of parameters and/or groups
 *        that is repeatable as a whole
 *        Note that a repeatable group consisting entirely of non-blocking
 *        children is equivalent to a non-repeatable group of
 *        repeatable children.
 *
 *****************************************************************************/
template<class P2, class... Ps>
inline group
repeatable(parameter p1, P2 p2, Ps... ps)
{
    return group{std::move(p1), std::move(p2),
                 std::move(ps)...}.repeatable(true);
}

template<class P2, class... Ps>
inline group
repeatable(group p1, P2 p2, Ps... ps)
{
    return group{std::move(p1), std::move(p2),
                 std::move(ps)...}.repeatable(true);
}



/*************************************************************************//**
 *
 * @brief makes a parameter greedy (match with top priority)
 *
 *****************************************************************************/
inline parameter
greedy(parameter p) {
    return p.greedy(true);
}

inline parameter
operator ! (parameter p) {
    return greedy(p);
}



/*************************************************************************//**
 *
 * @brief recursively prepends a prefix to all flags
 *
 *****************************************************************************/
inline parameter&&
with_prefix(const arg_string& prefix, parameter&& p) {
    return std::move(with_prefix(prefix, p));
}


//-------------------------------------------------------------------
inline group&
with_prefix(const arg_string& prefix, group& g)
{
    for(auto& p : g) {
        if(p.is_group()) {
            with_prefix(prefix, p.as_group());
        } else {
            with_prefix(prefix, p.as_param());
        }
    }
    return g;
}


inline group&&
with_prefix(const arg_string& prefix, group&& params)
{
    return std::move(with_prefix(prefix, params));
}


template<class Param, class... Params>
inline group
with_prefix(arg_string prefix, Param&& param, Params&&... params)
{
    return with_prefix(prefix, group{std::forward<Param>(param),
                                     std::forward<Params>(params)...});
}



/*************************************************************************//**
 *
 * @brief recursively prepends a prefix to all flags
 *
 * @param shortpfx : used for single-letter flags
 * @param longpfx  : used for flags with length > 1
 *
 *****************************************************************************/
inline parameter&&
with_prefixes_short_long(const arg_string& shortpfx, const arg_string& longpfx,
                         parameter&& p)
{
    return std::move(with_prefixes_short_long(shortpfx, longpfx, p));
}


//-------------------------------------------------------------------
inline group&
with_prefixes_short_long(const arg_string& shortFlagPrefix,
                         const arg_string& longFlagPrefix,
                         group& g)
{
    for(auto& p : g) {
        if(p.is_group()) {
            with_prefixes_short_long(shortFlagPrefix, longFlagPrefix, p.as_group());
        } else {
            with_prefixes_short_long(shortFlagPrefix, longFlagPrefix, p.as_param());
        }
    }
    return g;
}


inline group&&
with_prefixes_short_long(const arg_string& shortFlagPrefix,
                         const arg_string& longFlagPrefix,
                         group&& params)
{
    return std::move(with_prefixes_short_long(shortFlagPrefix, longFlagPrefix,
                                              params));
}


template<class Param, class... Params>
inline group
with_prefixes_short_long(const arg_string& shortFlagPrefix,
                         const arg_string& longFlagPrefix,
                         Param&& param, Params&&... params)
{
    return with_prefixes_short_long(shortFlagPrefix, longFlagPrefix,
                                    group{std::forward<Param>(param),
                                          std::forward<Params>(params)...});
}








/*************************************************************************//**
 *
 * @brief parsing implementation details
 *
 *****************************************************************************/

namespace detail {


/*************************************************************************//**
 *
 * @brief DFS traverser that keeps track of 'scopes'
 *        scope = all parameters that are either bounded by
 *        two blocking parameters on the same depth level
 *        or the beginning/end of the outermost group
 *
 *****************************************************************************/
class scoped_dfs_traverser
{
public:
    using dfs_traverser = group::depth_first_traverser;

    scoped_dfs_traverser() = default;

    explicit
    scoped_dfs_traverser(const group& g):
        pos_{g}, lastMatch_{}, posAfterLastMatch_{}, scopes_{},
        curMatched_{false}, ignoreBlocks_{false},
        repeatGroupStarted_{false}, repeatGroupContinues_{false}
    {}

    const dfs_traverser& base() const noexcept { return pos_; }
    const dfs_traverser& last_match() const noexcept { return lastMatch_; }

    const group& parent() const noexcept { return pos_.parent(); }
    const group* repeat_group() const noexcept { return pos_.repeat_group(); }
    const group* join_group() const noexcept { return pos_.join_group(); }

    const pattern* operator ->() const noexcept { return pos_.operator->(); }
    const pattern& operator *() const noexcept { return *pos_; }

    const pattern* ptr() const noexcept { return pos_.operator->(); }

    explicit operator bool() const noexcept { return bool(pos_); }

    bool joinable() const noexcept { return pos_.joinable(); }
    arg_string common_flag_prefix() const { return pos_.common_flag_prefix(); }

    void ignore_blocking(bool yes) { ignoreBlocks_ = yes; }

    void invalidate() { pos_.invalidate(); curMatched_ = false; }
    bool matched() const noexcept { return curMatched_; }

    bool start_of_repeat_group() const noexcept { return repeatGroupStarted_; }

    //-----------------------------------------------------
    scoped_dfs_traverser&
    next_sibling() { pos_.next_sibling(); return *this; }

    scoped_dfs_traverser&
    next_alternative() { pos_.next_alternative(); return *this; }

    scoped_dfs_traverser&
    next_after_siblings() { pos_.next_after_siblings(); return *this; }

    //-----------------------------------------------------
    scoped_dfs_traverser&
    operator ++ ()
    {
        if(!pos_) return *this;

        if(pos_.is_last_in_path()) {
            return_to_outermost_scope();
            return *this;
        }

        //current pattern can block if it didn't match already
        if(!ignoreBlocks_ && !matched()) {
            //current group can block if we didn't have any match in it
            if(pos_.is_last_in_group() && pos_.parent().blocking()
                && (!posAfterLastMatch_ || &(posAfterLastMatch_.parent()) != &(pos_.parent())))
            {
                //ascend to parent's level
                ++pos_;
                //skip all siblings of parent group
                pos_.next_after_siblings();
                if(!pos_) return_to_outermost_scope();
            }
            else if(pos_->blocking() && !pos_->is_group()) {
                if(pos_.parent().exclusive()) { //is_alternative(pos_.level())) {
                    pos_.next_alternative();
                } else {
                    //no match => skip siblings of blocking param
                    pos_.next_after_siblings();
                }
                if(!pos_) return_to_outermost_scope();
            } else {
                ++pos_;
            }
        } else {
            ++pos_;
        }
        check_if_left_scope();
        return *this;
    }

    //-----------------------------------------------------
    void next_after_match(scoped_dfs_traverser match)
    {
        if(!match || ignoreBlocks_) return;

        check_repeat_group_start(match);

        lastMatch_ = match.base();

        if(!match->blocking() && match.base().parent().blocking()) {
            match.pos_.back_to_parent();
        }

        //if match is not in current position & current position is blocking
        //=> current position has to be advanced by one so that it is
        //no longer reachable within current scope
        //(can happen for repeatable, blocking parameters)
        if(match.base() != pos_ && pos_->blocking()) pos_.next_sibling();

        if(match->blocking()) {
            if(match.pos_.is_alternative()) {
                //discard other alternatives
                match.pos_.skip_alternatives();
            }

            if(is_last_in_current_scope(match.pos_)) {
                //if current param is not repeatable -> back to previous scope
                if(!match->repeatable() && !match->is_group()) {
                    curMatched_ = false;
                    pos_ = std::move(match.pos_);
                    if(!scopes_.empty()) pos_.undo(scopes_.top());
                }
                else { //stay at match position
                    curMatched_ = true;
                    pos_ = std::move(match.pos_);
                }
            }
            else { //not last in current group
                //if current param is not repeatable, go directly to next
                if(!match->repeatable() && !match->is_group()) {
                    curMatched_ = false;
                    ++match.pos_;
                } else {
                    curMatched_ = true;
                }

                if(match.pos_.level() > pos_.level()) {
                    scopes_.push(pos_.undo_point());
                    pos_ = std::move(match.pos_);
                }
                else if(match.pos_.level() < pos_.level()) {
                    return_to_level(match.pos_.level());
                }
                else {
                    pos_ = std::move(match.pos_);
                }
            }
            posAfterLastMatch_ = pos_;
        }
        else {
            if(match.pos_.level() < pos_.level()) {
                return_to_level(match.pos_.level());
            }
            posAfterLastMatch_ = pos_;
        }
        repeatGroupContinues_ = repeat_group_continues();
    }

private:
    //-----------------------------------------------------
    bool is_last_in_current_scope(const dfs_traverser& pos)
    {
        if(scopes_.empty()) return pos.is_last_in_path();
        //check if we would leave the current scope on ++
        auto p = pos;
        ++p;
        return p.level() < scopes_.top().level();
    }

    //-----------------------------------------------------
    void check_repeat_group_start(const scoped_dfs_traverser& newMatch)
    {
        const auto newrg = newMatch.repeat_group();
        if(!newrg) {
            repeatGroupStarted_ = false;
        }
        else if(lastMatch_.repeat_group() != newrg) {
            repeatGroupStarted_ = true;
        }
        else if(!repeatGroupContinues_ || !newMatch.repeatGroupContinues_) {
            repeatGroupStarted_ = true;
        }
        else {
            //special case: repeat group is outermost group
            //=> we can never really 'leave' and 'reenter' it
            //but if the current scope is the first element, then we are
            //conceptually at a position 'before' the group
            repeatGroupStarted_ = scopes_.empty() || (
                    newrg == pos_.root() &&
                    scopes_.top().param() == &(*pos_.root()->begin()) );
        }
        repeatGroupContinues_ = repeatGroupStarted_;
    }

    //-----------------------------------------------------
    bool repeat_group_continues()
    {
        if(!repeatGroupContinues_) return false;
        const auto curRepGroup = pos_.repeat_group();
        if(!curRepGroup) return false;
        if(curRepGroup != lastMatch_.repeat_group()) return false;
        if(!posAfterLastMatch_) return false;
        return true;
    }

    //-----------------------------------------------------
    void check_if_left_scope()
    {
        if(posAfterLastMatch_) {
            if(pos_.level() < posAfterLastMatch_.level()) {
                while(!scopes_.empty() && scopes_.top().level() >= pos_.level()) {
                    pos_.undo(scopes_.top());
                    scopes_.pop();
                }
                posAfterLastMatch_.invalidate();
            }
        }
        while(!scopes_.empty() && scopes_.top().level() > pos_.level()) {
            pos_.undo(scopes_.top());
            scopes_.pop();
        }
        repeatGroupContinues_ = repeat_group_continues();
    }

    //-----------------------------------------------------
    void return_to_outermost_scope()
    {
        posAfterLastMatch_.invalidate();

        if(scopes_.empty()) {
            pos_.invalidate();
            repeatGroupContinues_ = false;
            return;
        }

        while(!scopes_.empty() && (!pos_ || pos_.level() >= 1)) {
            pos_.undo(scopes_.top());
            scopes_.pop();
        }
        while(!scopes_.empty()) scopes_.pop();

        repeatGroupContinues_ = repeat_group_continues();
    }

    //-----------------------------------------------------
    void return_to_level(int level)
    {
        if(pos_.level() <= level) return;
        while(!scopes_.empty() && pos_.level() > level) {
            pos_.undo(scopes_.top());
            scopes_.pop();
        }
    };

    dfs_traverser pos_;
    dfs_traverser lastMatch_;
    dfs_traverser posAfterLastMatch_;
    std::stack<dfs_traverser::memento> scopes_;
    bool curMatched_ = false;
    bool ignoreBlocks_ = false;
    bool repeatGroupStarted_ = false;
    bool repeatGroupContinues_ = false;
};




/*****************************************************************************
 *
 * some parameter property predicates
 *
 *****************************************************************************/
struct select_all {
    bool operator () (const parameter&) const noexcept { return true; }
};

struct select_flags {
    bool operator () (const parameter& p) const noexcept {
        return !p.flags().empty();
    }
};

struct select_values {
    bool operator () (const parameter& p) const noexcept {
        return p.flags().empty();
    }
};



/*************************************************************************//**
 *
 * @brief result of a matching operation
 *
 *****************************************************************************/
class match_t {
public:
    match_t() = default;
    match_t(arg_string s, scoped_dfs_traverser p):
        str_{std::move(s)}, pos_{std::move(p)}
    {}

    const arg_string& str() const noexcept { return str_; }
    const scoped_dfs_traverser& pos() const noexcept { return pos_; }

    explicit operator bool() const noexcept { return !str_.empty(); }

private:
    arg_string str_;
    scoped_dfs_traverser pos_;
};



/*************************************************************************//**
 *
 * @brief finds the first parameter that matches a given string;
 *        candidate parameters are traversed using a scoped DFS traverser
 *
 *****************************************************************************/
template<class ParamSelector>
match_t
full_match(scoped_dfs_traverser pos, const arg_string& arg,
           const ParamSelector& select)
{
    if(arg.empty()) return match_t{};

    while(pos) {
        if(pos->is_param()) {
            const auto& param = pos->as_param();
            if(select(param)) {
                const auto match = param.match(arg);
                if(match && match.length() == arg.size()) {
                    return match_t{arg, std::move(pos)};
                }
            }
        }
        ++pos;
    }
    return match_t{};
}



/*************************************************************************//**
 *
 * @brief finds the first parameter that matches any (non-empty) prefix
 *        of a given string;
 *        candidate parameters are traversed using a scoped DFS traverser
 *
 *****************************************************************************/
template<class ParamSelector>
match_t
prefix_match(scoped_dfs_traverser pos, const arg_string& arg,
             const ParamSelector& select)
{
    if(arg.empty()) return match_t{};

    while(pos) {
        if(pos->is_param()) {
            const auto& param = pos->as_param();
            if(select(param)) {
                const auto match = param.match(arg);
                if(match.prefix()) {
                    if(match.length() == arg.size()) {
                        return match_t{arg, std::move(pos)};
                    }
                    else {
                        return match_t{arg.substr(match.at(), match.length()),
                                       std::move(pos)};
                    }
                }
            }
        }
        ++pos;
    }
    return match_t{};
}



/*************************************************************************//**
 *
 * @brief finds the first parameter that partially matches a given string;
 *        candidate parameters are traversed using a scoped DFS traverser
 *
 *****************************************************************************/
template<class ParamSelector>
match_t
partial_match(scoped_dfs_traverser pos, const arg_string& arg,
              const ParamSelector& select)
{
    if(arg.empty()) return match_t{};

    while(pos) {
        if(pos->is_param()) {
            const auto& param = pos->as_param();
            if(select(param)) {
                const auto match = param.match(arg);
                if(match) {
                    return match_t{arg.substr(match.at(), match.length()),
                                   std::move(pos)};
                }
            }
        }
        ++pos;
    }
    return match_t{};
}

} //namespace detail






/***************************************************************//**
 *
 * @brief default command line arguments parser
 *
 *******************************************************************/
class parser
{
public:
    using dfs_traverser = group::depth_first_traverser;
    using scoped_dfs_traverser = detail::scoped_dfs_traverser;


    /*****************************************************//**
     * @brief arg -> parameter mapping
     *********************************************************/
    class arg_mapping {
    public:
        friend class parser;

        explicit
        arg_mapping(arg_index idx, arg_string s,
                    const dfs_traverser& match)
        :
            index_{idx}, arg_{std::move(s)}, match_{match},
            repeat_{0}, startsRepeatGroup_{false},
            blocked_{false}, conflict_{false}
        {}

        explicit
        arg_mapping(arg_index idx, arg_string s) :
            index_{idx}, arg_{std::move(s)}, match_{},
            repeat_{0}, startsRepeatGroup_{false},
            blocked_{false}, conflict_{false}
        {}

        arg_index index() const noexcept { return index_; }
        const arg_string& arg() const noexcept { return arg_; }

        const parameter* param() const noexcept {
            return match_ && match_->is_param()
                ? &(match_->as_param()) : nullptr;
        }

        std::size_t repeat() const noexcept { return repeat_; }

        bool blocked() const noexcept { return blocked_; }
        bool conflict() const noexcept { return conflict_; }

        bool bad_repeat() const noexcept {
            if(!param()) return false;
            return repeat_ > 0 && !param()->repeatable()
                && !match_.repeat_group();
        }

        bool any_error() const noexcept {
            return !match_ || blocked() || conflict() || bad_repeat();
        }

    private:
        arg_index index_;
        arg_string arg_;
        dfs_traverser match_;
        std::size_t repeat_;
        bool startsRepeatGroup_;
        bool blocked_;
        bool conflict_;
    };

    /*****************************************************//**
     * @brief references a non-matched, required parameter
     *********************************************************/
    class missing_event {
    public:
        explicit
        missing_event(const parameter* p, arg_index after):
            param_{p}, aftIndex_{after}
        {}

        const parameter* param() const noexcept { return param_; }

        arg_index after_index() const noexcept { return aftIndex_; }

    private:
        const parameter* param_;
        arg_index aftIndex_;
    };

    //-----------------------------------------------------
    using missing_events = std::vector<missing_event>;
    using arg_mappings   = std::vector<arg_mapping>;


private:
    struct miss_candidate {
        miss_candidate(dfs_traverser p, arg_index idx,
                       bool firstInRepeatGroup = false):
            pos{std::move(p)}, index{idx},
            startsRepeatGroup{firstInRepeatGroup}
        {}

        dfs_traverser pos;
        arg_index index;
        bool startsRepeatGroup;
    };
    using miss_candidates = std::vector<miss_candidate>;


public:
    //---------------------------------------------------------------
    /** @brief initializes parser with a command line interface
     *  @param offset = argument index offset used for reports
     * */
    explicit
    parser(const group& root, arg_index offset = 0):
        root_{&root}, pos_{root},
        index_{offset-1}, eaten_{0},
        args_{}, missCand_{}, blocked_{false}
    {
        for_each_potential_miss(dfs_traverser{root},
            [this](const dfs_traverser& p){
                missCand_.emplace_back(p, index_);
            });
    }


    //---------------------------------------------------------------
    /** @brief processes one command line argument */
    bool operator() (const arg_string& arg)
    {
        ++eaten_;
        ++index_;

        if(!valid() || arg.empty()) return false;

        if(!blocked_ && try_match(arg)) return true;

        if(try_match_blocked(arg)) return false;

        //skipping of blocking & required patterns is not allowed
        if(!blocked_ && !pos_.matched() && pos_->required() && pos_->blocking()) {
            blocked_ = true;
        }

        add_nomatch(arg);
        return false;
    }


    //---------------------------------------------------------------
    /** @brief returns range of argument -> parameter mappings */
    const arg_mappings& args() const {
        return args_;
    }

    /** @brief returns list of missing events */
    missing_events missed() const {
        missing_events misses;
        misses.reserve(missCand_.size());
        for(auto i = missCand_.begin(); i != missCand_.end(); ++i) {
            misses.emplace_back(&(i->pos->as_param()), i->index);
        }
        return misses;
    }

    /** @brief returns number of processed command line arguments */
    arg_index parse_count() const noexcept { return eaten_; }

    /** @brief returns false if previously processed command line arguments
     *         lead to an invalid / inconsistent parsing result
     */
    bool valid() const noexcept { return bool(pos_); }

    /** @brief returns false if previously processed command line arguments
     *         lead to an invalid / inconsistent parsing result
     */
    explicit operator bool() const noexcept { return valid(); }


private:
    //---------------------------------------------------------------
    using match_t = detail::match_t;


    //---------------------------------------------------------------
    /** @brief try to match argument with unreachable parameter */
    bool try_match_blocked(const arg_string& arg)
    {
        //try to match ahead (using temporary parser)
        if(pos_) {
            auto ahead = *this;
            if(try_match_blocked(std::move(ahead), arg)) return true;
        }

        //try to match from the beginning (using temporary parser)
        if(root_) {
            parser all{*root_, index_+1};
            if(try_match_blocked(std::move(all), arg)) return true;
        }

        return false;
    }

    //---------------------------------------------------------------
    bool try_match_blocked(parser&& parse, const arg_string& arg)
    {
        const auto nold = int(parse.args_.size());

        parse.pos_.ignore_blocking(true);

        if(!parse.try_match(arg)) return false;

        for(auto i = parse.args_.begin() + nold; i != parse.args_.end(); ++i) {
            args_.push_back(*i);
            args_.back().blocked_ = true;
        }
        return true;
    }

    //---------------------------------------------------------------
    /** @brief try to find a parameter/pattern that matches 'arg' */
    bool try_match(const arg_string& arg)
    {
        //match greedy parameters before everything else
        if(pos_->is_param() && pos_->blocking() && pos_->as_param().greedy()) {
            const auto match = pos_->as_param().match(arg);
            if(match && match.length() == arg.size()) {
                add_match(detail::match_t{arg,pos_});
                return true;
            }
        }

        //try flags first (alone, joinable or strict sequence)
        if(try_match_full(arg, detail::select_flags{})) return true;
        if(try_match_joined_flags(arg)) return true;
        if(try_match_joined_sequence(arg, detail::select_flags{})) return true;
        //try value params (alone or strict sequence)
        if(try_match_full(arg, detail::select_values{})) return true;
        if(try_match_joined_sequence(arg, detail::select_all{})) return true;
        //try joinable params + values in any order
        if(try_match_joined_params(arg)) return true;
        return false;
    }

    //---------------------------------------------------------------
    /**
     * @brief try to match full argument
     * @param select : predicate that candidate parameters must satisfy
     */
    template<class ParamSelector>
    bool try_match_full(const arg_string& arg, const ParamSelector& select)
    {
        auto match = detail::full_match(pos_, arg, select);
        if(!match) return false;
        add_match(match);
        return true;
    }

    //---------------------------------------------------------------
    /**
     * @brief try to match argument as blocking sequence of parameters
     * @param select : predicate that a parameter matching the prefix of
     *                 'arg' must satisfy
     */
    template<class ParamSelector>
    bool try_match_joined_sequence(arg_string arg,
                                   const ParamSelector& acceptFirst)
    {
        auto fstMatch = detail::prefix_match(pos_, arg, acceptFirst);

        if(!fstMatch) return false;

        if(fstMatch.str().size() == arg.size()) {
            add_match(fstMatch);
            return true;
        }

        if(!fstMatch.pos()->blocking()) return false;

        auto pos = fstMatch.pos();
        pos.ignore_blocking(true);
        const auto parent = &pos.parent();
        if(!pos->repeatable()) ++pos;

        arg.erase(0, fstMatch.str().size());
        std::vector<match_t> matches { std::move(fstMatch) };

        while(!arg.empty() && pos &&
              pos->blocking() && pos->is_param() &&
              (&pos.parent() == parent))
        {
            auto match = pos->as_param().match(arg);

            if(match.prefix()) {
                matches.emplace_back(arg.substr(0,match.length()), pos);
                arg.erase(0, match.length());
                if(!pos->repeatable()) ++pos;
            }
            else {
                if(!pos->repeatable()) return false;
                ++pos;
            }

        }
        //if arg not fully covered => discard temporary matches
        if(!arg.empty() || matches.empty()) return false;

        for(const auto& m : matches) add_match(m);
        return true;
    }

    //-----------------------------------------------------
    /** @brief try to match 'arg' as a concatenation of joinable flags */
    bool try_match_joined_flags(const arg_string& arg)
    {
        return find_join_group(pos_, [&](const group& g) {
            return try_match_joined(g, arg, detail::select_flags{},
                                    g.common_flag_prefix());
        });
    }

    //---------------------------------------------------------------
    /** @brief try to match 'arg' as a concatenation of joinable parameters */
    bool try_match_joined_params(const arg_string& arg)
    {
        return find_join_group(pos_, [&](const group& g) {
            return try_match_joined(g, arg, detail::select_all{});
        });
    }

    //-----------------------------------------------------
    /** @brief try to match 'arg' as concatenation of joinable parameters
     *         that are all contaied within one group
     */
    template<class ParamSelector>
    bool try_match_joined(const group& joinGroup, arg_string arg,
                          const ParamSelector& select,
                          const arg_string& prefix = "")
    {
        //temporary parser with 'joinGroup' as top-level group
        parser parse {joinGroup};
        //records temporary matches
        std::vector<match_t> matches;

        while(!arg.empty()) {
            auto match = detail::prefix_match(parse.pos_, arg, select);

            if(!match) return false;

            arg.erase(0, match.str().size());
            //make sure prefix is always present after the first match
            //so that, e.g., flags "-a" and "-b" will be found in "-ab"
            if(!arg.empty() && !prefix.empty() && arg.find(prefix) != 0 &&
                prefix != match.str())
            {
                arg.insert(0,prefix);
            }

            parse.add_match(match);
            matches.push_back(std::move(match));
        }

        if(!arg.empty() || matches.empty()) return false;

        if(!parse.missCand_.empty()) return false;
        for(const auto& a : parse.args_) if(a.any_error()) return false;

        //replay matches onto *this
        for(const auto& m : matches) add_match(m);
        return true;
    }

    //-----------------------------------------------------
    template<class GroupSelector>
    bool find_join_group(const scoped_dfs_traverser& start,
                         const GroupSelector& accept) const
    {
        if(start && start.parent().joinable()) {
            const auto& g = start.parent();
            if(accept(g)) return true;
            return false;
        }

        auto pos = start;
        while(pos) {
            if(pos->is_group() && pos->as_group().joinable()) {
                const auto& g = pos->as_group();
                if(accept(g)) return true;
                pos.next_sibling();
            }
            else {
                ++pos;
            }
        }
        return false;
    }


    //---------------------------------------------------------------
    void add_nomatch(const arg_string& arg) {
        args_.emplace_back(index_, arg);
    }


    //---------------------------------------------------------------
    void add_match(const match_t& match)
    {
        const auto& pos = match.pos();
        if(!pos || !pos->is_param() || match.str().empty()) return;

        pos_.next_after_match(pos);

        arg_mapping newArg{index_, match.str(), pos.base()};
        newArg.repeat_ = occurrences_of(&pos->as_param());
        newArg.conflict_ = check_conflicts(pos.base());
        newArg.startsRepeatGroup_ = pos_.start_of_repeat_group();
        args_.push_back(std::move(newArg));

        add_miss_candidates_after(pos);
        clean_miss_candidates_for(pos.base());
        discard_alternative_miss_candidates(pos.base());

    }

    //-----------------------------------------------------
    bool check_conflicts(const dfs_traverser& match)
    {
        if(pos_.start_of_repeat_group()) return false;
        bool conflict = false;
        for(const auto& m : match.stack()) {
            if(m.parent->exclusive()) {
                for(auto i = args_.rbegin(); i != args_.rend(); ++i) {
                    if(!i->blocked()) {
                        for(const auto& c : i->match_.stack()) {
                            //sibling within same exclusive group => conflict
                            if(c.parent == m.parent && c.cur != m.cur) {
                                conflict = true;
                                i->conflict_ = true;
                            }
                        }
                    }
                    //check for conflicts only within current repeat cycle
                    if(i->startsRepeatGroup_) break;
                }
            }
        }
        return conflict;
    }

    //-----------------------------------------------------
    void clean_miss_candidates_for(const dfs_traverser& match)
    {
        auto i = std::find_if(missCand_.rbegin(), missCand_.rend(),
            [&](const miss_candidate& m) {
                return &(*m.pos) == &(*match);
            });

        if(i != missCand_.rend()) {
            missCand_.erase(prev(i.base()));
        }
    }

    //-----------------------------------------------------
    void discard_alternative_miss_candidates(const dfs_traverser& match)
    {
        if(missCand_.empty()) return;
        //find out, if miss candidate is sibling of one of the same
        //alternative groups that the current match is a member of
        //if so, we can discard the miss

        //go through all exclusive groups of matching pattern
        for(const auto& m : match.stack()) {
            if(m.parent->exclusive()) {
                for(auto i = int(missCand_.size())-1; i >= 0; --i) {
                    bool removed = false;
                    for(const auto& c : missCand_[i].pos.stack()) {
                        //sibling within same exclusive group => discard
                        if(c.parent == m.parent && c.cur != m.cur) {
                            missCand_.erase(missCand_.begin() + i);
                            if(missCand_.empty()) return;
                            removed = true;
                            break;
                        }
                    }
                    //remove miss candidates only within current repeat cycle
                    if(i > 0 && removed) {
                        if(missCand_[i-1].startsRepeatGroup) break;
                    } else {
                        if(missCand_[i].startsRepeatGroup) break;
                    }
                }
            }
        }
    }

    //-----------------------------------------------------
    void add_miss_candidates_after(const scoped_dfs_traverser& match)
    {
        auto npos = match.base();
        if(npos.is_alternative()) npos.skip_alternatives();
        ++npos;
        //need to add potential misses if:
        //either new repeat group was started
        const auto newRepGroup = match.repeat_group();
        if(newRepGroup) {
            if(pos_.start_of_repeat_group()) {
                for_each_potential_miss(std::move(npos),
                    [&,this](const dfs_traverser& pos) {
                        //only add candidates within repeat group
                        if(newRepGroup == pos.repeat_group()) {
                            missCand_.emplace_back(pos, index_, true);
                        }
                    });
            }
        }
        //... or an optional blocking param was hit
        else if(match->blocking() && !match->required() &&
            npos.level() >= match.base().level())
        {
            for_each_potential_miss(std::move(npos),
                [&,this](const dfs_traverser& pos) {
                    //only add new candidates
                    if(std::find_if(missCand_.begin(), missCand_.end(),
                        [&](const miss_candidate& c){
                            return &(*c.pos) == &(*pos);
                        }) == missCand_.end())
                    {
                        missCand_.emplace_back(pos, index_);
                    }
                });
        }

    }

    //-----------------------------------------------------
    template<class Action>
    static void
    for_each_potential_miss(dfs_traverser pos, Action&& action)
    {
        const auto level = pos.level();
        while(pos && pos.level() >= level) {
            if(pos->is_group() ) {
                const auto& g = pos->as_group();
                if(g.all_optional() || (g.exclusive() && g.any_optional())) {
                    pos.next_sibling();
                } else {
                    ++pos;
                }
            } else {  //param
                if(pos->required()) {
                    action(pos);
                    ++pos;
                } else if(pos->blocking()) { //optional + blocking
                    pos.next_after_siblings();
                } else {
                    ++pos;
                }
            }
        }
    }


    //---------------------------------------------------------------
    std::size_t occurrences_of(const parameter* p) const
    {
        auto i = std::find_if(args_.rbegin(), args_.rend(),
            [p](const arg_mapping& a){ return a.param() == p; });

        if(i != args_.rend()) return i->repeat() + 1;
        return 0;
    }


    //---------------------------------------------------------------
    const group* root_;
    scoped_dfs_traverser pos_;
    arg_index index_;
    arg_index eaten_;
    arg_mappings args_;
    miss_candidates missCand_;
    bool blocked_;
};




/*************************************************************************//**
 *
 * @brief contains argument -> parameter mappings
 *        and missing parameters
 *
 *****************************************************************************/
class parsing_result
{
public:
    using arg_mapping    = parser::arg_mapping;
    using arg_mappings   = parser::arg_mappings;
    using missing_event  = parser::missing_event;
    using missing_events = parser::missing_events;
    using iterator       = arg_mappings::const_iterator;

    //-----------------------------------------------------
    /** @brief default: empty redult */
    parsing_result() = default;

    parsing_result(arg_mappings arg2param, missing_events misses):
        arg2param_{std::move(arg2param)}, missing_{std::move(misses)}
    {}

    //-----------------------------------------------------
    /** @brief returns number of arguments that could not be mapped to
     *         a parameter
     */
    arg_mappings::size_type
    unmapped_args_count() const noexcept {
        return std::count_if(arg2param_.begin(), arg2param_.end(),
            [](const arg_mapping& a){ return !a.param(); });
    }

    /** @brief returns if any argument could only be matched by an
     *         unreachable parameter
     */
    bool any_blocked() const noexcept {
        return std::any_of(arg2param_.begin(), arg2param_.end(),
            [](const arg_mapping& a){ return a.blocked(); });
    }

    /** @brief returns if any argument matched more than one parameter
     *         that were mutually exclusive */
    bool any_conflict() const noexcept {
        return std::any_of(arg2param_.begin(), arg2param_.end(),
            [](const arg_mapping& a){ return a.conflict(); });
    }

    /** @brief returns if any parameter matched repeatedly although
     *         it was not allowed to */
    bool any_bad_repeat() const noexcept {
        return std::any_of(arg2param_.begin(), arg2param_.end(),
            [](const arg_mapping& a){ return a.bad_repeat(); });
    }

    /** @brief returns true if any parsing error / violation of the
     *         command line interface definition occured */
    bool any_error() const noexcept {
        return unmapped_args_count() > 0 || !missing().empty() ||
               any_blocked() || any_conflict() || any_bad_repeat();
    }

    /** @brief returns true if no parsing error / violation of the
     *         command line interface definition occured */
    explicit operator bool() const noexcept { return !any_error(); }

    /** @brief access to range of missing parameter match events */
    const missing_events& missing() const noexcept { return missing_; }

    /** @brief returns non-mutating iterator to position of
     *         first argument -> parameter mapping  */
    iterator begin() const noexcept { return arg2param_.begin(); }
    /** @brief returns non-mutating iterator to position one past the
     *         last argument -> parameter mapping  */
    iterator end()   const noexcept { return arg2param_.end(); }

private:
    //-----------------------------------------------------
    arg_mappings arg2param_;
    missing_events missing_;
};




namespace detail {
namespace {

/*************************************************************************//**
 *
 * @brief correct some common problems
 *        does not - and MUST NOT - change the number of arguments
 *        (no insertions or deletions allowed)
 *
 *****************************************************************************/
void sanitize_args(arg_list& args)
{
    //e.g. {"-o12", ".34"} -> {"-o", "12.34"}

    if(args.empty()) return;

    for(auto i = begin(args)+1; i != end(args); ++i) {
        if(i != begin(args) && i->size() > 1 &&
            i->find('.') == 0 && std::isdigit((*i)[1]) )
        {
            //find trailing digits in previous arg
            using std::prev;
            auto& prv = *prev(i);
            auto fstDigit = std::find_if_not(prv.rbegin(), prv.rend(),
                [](arg_string::value_type c){
                    return std::isdigit(c);
                }).base();

            //handle leading sign
            if(fstDigit > prv.begin() &&
                (*prev(fstDigit) == '+' || *prev(fstDigit) == '-'))
            {
                --fstDigit;
            }

            //prepend digits from previous arg
            i->insert(begin(*i), fstDigit, end(prv));

            //erase digits in previous arg
            prv.erase(fstDigit, end(prv));
        }
    }
}



/*************************************************************************//**
 *
 * @brief executes actions based on a parsing result
 *
 *****************************************************************************/
void execute_actions(const parsing_result& res)
{
    for(const auto& m : res) {
        if(m.param()) {
            const auto& param = *(m.param());

            if(m.repeat() > 0) param.notify_repeated(m.index());
            if(m.blocked())    param.notify_blocked(m.index());
            if(m.conflict())   param.notify_conflict(m.index());
            //main action
            if(!m.any_error()) param.execute_actions(m.arg());
        }
    }

    for(auto m : res.missing()) {
        if(m.param()) m.param()->notify_missing(m.after_index());
    }
}



/*************************************************************************//**
 *
 * @brief parses input args
 *
 *****************************************************************************/
static parsing_result
parse_args(const arg_list& args, const group& cli,
           arg_index offset = 0)
{
    //parse args and store unrecognized arg indices
    parser parse{cli, offset};
    for(const auto& arg : args) {
        parse(arg);
        if(!parse.valid()) break;
    }

    return parsing_result{parse.args(), parse.missed()};
}

/*************************************************************************//**
 *
 * @brief parses input args & executes actions
 *
 *****************************************************************************/
static parsing_result
parse_and_execute(const arg_list& args, const group& cli,
                  arg_index offset = 0)
{
    auto result = parse_args(args, cli, offset);

    execute_actions(result);

    return result;
}

} //anonymous namespace
} // namespace detail




/*************************************************************************//**
 *
 * @brief parses vector of arg strings and executes actions
 *
 *****************************************************************************/
inline parsing_result
parse(arg_list args, const group& cli)
{
    detail::sanitize_args(args);
    return detail::parse_and_execute(args, cli);
}


/*************************************************************************//**
 *
 * @brief parses initializer_list of C-style arg strings and executes actions
 *
 *****************************************************************************/
inline parsing_result
parse(std::initializer_list<const char*> arglist, const group& cli)
{
    arg_list args;
    args.reserve(arglist.size());
    for(auto a : arglist) {
        if(std::strlen(a) > 0) args.push_back(a);
    }

    return parse(std::move(args), cli);
}


/*************************************************************************//**
 *
 * @brief parses range of arg strings and executes actions
 *
 *****************************************************************************/
template<class InputIterator>
inline parsing_result
parse(InputIterator first, InputIterator last, const group& cli)
{
    return parse(arg_list(first,last), cli);
}


/*************************************************************************//**
 *
 * @brief parses the standard array of command line arguments; omits argv[0]
 *
 *****************************************************************************/
inline parsing_result
parse(const int argc, char* argv[], const group& cli, arg_index offset = 1)
{
    arg_list args;
    if(offset < argc) args.assign(argv+offset, argv+argc);
    detail::sanitize_args(args);
    return detail::parse_and_execute(args, cli, offset);
}






/*************************************************************************//**
 *
 * @brief filter predicate for parameters and groups;
 *        Can be used to limit documentation generation to parameter subsets.
 *
 *****************************************************************************/
class param_filter
{
public:
    /** @brief only allow parameters with given prefix */
    param_filter& prefix(const arg_string& p) noexcept {
        prefix_ = p; return *this;
    }
    /** @brief only allow parameters with given prefix */
    param_filter& prefix(arg_string&& p) noexcept {
        prefix_ = std::move(p); return *this;
    }
    const arg_string& prefix()  const noexcept { return prefix_; }

    /** @brief only allow parameters with given requirement status */
    param_filter& required(tri t)  noexcept { required_ = t; return *this; }
    tri           required() const noexcept { return required_; }

    /** @brief only allow parameters with given blocking status */
    param_filter& blocking(tri t)  noexcept { blocking_ = t; return *this; }
    tri           blocking() const noexcept { return blocking_; }

    /** @brief only allow parameters with given repeatable status */
    param_filter& repeatable(tri t)  noexcept { repeatable_ = t; return *this; }
    tri           repeatable() const noexcept { return repeatable_; }

    /** @brief only allow parameters with given docstring status */
    param_filter& has_doc(tri t)  noexcept { hasDoc_ = t; return *this; }
    tri           has_doc() const noexcept { return hasDoc_; }


    /** @brief returns true, if parameter satisfies all filters */
    bool operator() (const parameter& p) const noexcept {
        if(!prefix_.empty()) {
            if(!std::any_of(p.flags().begin(), p.flags().end(),
                [&](const arg_string& flag){
                    return str::has_prefix(flag, prefix_);
                })) return false;
        }
        if(required()   != p.required())     return false;
        if(blocking()   != p.blocking())     return false;
        if(repeatable() != p.repeatable())   return false;
        if(has_doc()    != !p.doc().empty()) return false;
        return true;
    }

private:
    arg_string prefix_;
    tri required_   = tri::either;
    tri blocking_   = tri::either;
    tri repeatable_ = tri::either;
    tri exclusive_  = tri::either;
    tri hasDoc_     = tri::yes;
};






/*************************************************************************//**
 *
 * @brief documentation formatting options
 *
 *****************************************************************************/
class doc_formatting
{
public:
    using string = doc_string;

    /** @brief same as 'first_column' */
#if __cplusplus >= 201402L
    [[deprecated]]
#endif
    doc_formatting& start_column(int col) { return first_column(col); }
#if __cplusplus >= 201402L
    [[deprecated]]
#endif
    int start_column() const noexcept { return first_column(); }

    /** @brief determines column where documentation printing starts */
    doc_formatting&
    first_column(int col) {
        //limit to [0,last_column] but push doc_column to the right if neccessary
        if(col < 0) col = 0;
        else if(col > last_column()) col = last_column();
        if(col > doc_column()) doc_column(first_column());
        firstCol_ = col;
        return *this;
    }
    int first_column() const noexcept {
        return firstCol_;
    }

    /** @brief determines column where docstrings start */
    doc_formatting&
    doc_column(int col) {
        //limit to [first_column,last_column]
        if(col < 0) col = 0;
        else if(col < first_column()) col = first_column();
        else if(col > last_column()) col = last_column();
        docCol_ = col;
        return *this;
    }
    int doc_column() const noexcept {
        return docCol_;
    }

    /** @brief determines column that no documentation text must exceed;
     *         (text should be wrapped appropriately after this column)
     */
    doc_formatting&
    last_column(int col) {
        //limit to [first_column,oo] but push doc_column to the left if neccessary
        if(col < first_column()) col = first_column();
        if(col < doc_column()) doc_column(col);
        lastCol_ = col;
        return *this;
    }

    int last_column() const noexcept {
        return lastCol_;
    }

    /** @brief determines indent of documentation lines
     *         for children of a documented group */
    doc_formatting& indent_size(int indent) { indentSize_ = indent; return *this; }
    int             indent_size() const noexcept  { return indentSize_; }

    /** @brief determines string to be used
     *         if a parameter has no flags and no label  */
    doc_formatting& empty_label(const string& label) {
        emptyLabel_ = label;
        return *this;
    }
    const string& empty_label() const noexcept { return emptyLabel_; }

    /** @brief determines string for separating parameters */
    doc_formatting& param_separator(const string& sep) {
        paramSep_ = sep;
        return *this;
    }
    const string& param_separator() const noexcept { return paramSep_; }

    /** @brief determines string for separating groups (in usage lines) */
    doc_formatting& group_separator(const string& sep) {
        groupSep_ = sep;
        return *this;
    }
    const string& group_separator() const noexcept { return groupSep_; }

    /** @brief determines string for separating alternative parameters */
    doc_formatting& alternative_param_separator(const string& sep) {
        altParamSep_ = sep;
        return *this;
    }
    const string& alternative_param_separator() const noexcept { return altParamSep_; }

    /** @brief determines string for separating alternative groups */
    doc_formatting& alternative_group_separator(const string& sep) {
        altGroupSep_ = sep;
        return *this;
    }
    const string& alternative_group_separator() const noexcept { return altGroupSep_; }

    /** @brief determines string for separating flags of the same parameter */
    doc_formatting& flag_separator(const string& sep) {
        flagSep_ = sep;
        return *this;
    }
    const string& flag_separator() const noexcept { return flagSep_; }

    /** @brief determnines strings surrounding parameter labels */
    doc_formatting&
    surround_labels(const string& prefix, const string& postfix) {
        labelPre_ = prefix;
        labelPst_ = postfix;
        return *this;
    }
    const string& label_prefix()  const noexcept { return labelPre_; }
    const string& label_postfix() const noexcept { return labelPst_; }

    /** @brief determnines strings surrounding optional parameters/groups */
    doc_formatting&
    surround_optional(const string& prefix, const string& postfix) {
        optionPre_ = prefix;
        optionPst_ = postfix;
        return *this;
    }
    const string& optional_prefix()  const noexcept { return optionPre_; }
    const string& optional_postfix() const noexcept { return optionPst_; }

    /** @brief determnines strings surrounding repeatable parameters/groups */
    doc_formatting&
    surround_repeat(const string& prefix, const string& postfix) {
        repeatPre_ = prefix;
        repeatPst_ = postfix;
        return *this;
    }
    const string& repeat_prefix()  const noexcept { return repeatPre_; }
    const string& repeat_postfix() const noexcept { return repeatPst_; }

    /** @brief determnines strings surrounding exclusive groups */
    doc_formatting&
    surround_alternatives(const string& prefix, const string& postfix) {
        alternPre_ = prefix;
        alternPst_ = postfix;
        return *this;
    }
    const string& alternatives_prefix()  const noexcept { return alternPre_; }
    const string& alternatives_postfix() const noexcept { return alternPst_; }

    /** @brief determnines strings surrounding alternative flags */
    doc_formatting&
    surround_alternative_flags(const string& prefix, const string& postfix) {
        alternFlagPre_ = prefix;
        alternFlagPst_ = postfix;
        return *this;
    }
    const string& alternative_flags_prefix()  const noexcept { return alternFlagPre_; }
    const string& alternative_flags_postfix() const noexcept { return alternFlagPst_; }

    /** @brief determnines strings surrounding non-exclusive groups */
    doc_formatting&
    surround_group(const string& prefix, const string& postfix) {
        groupPre_ = prefix;
        groupPst_ = postfix;
        return *this;
    }
    const string& group_prefix()  const noexcept { return groupPre_; }
    const string& group_postfix() const noexcept { return groupPst_; }

    /** @brief determnines strings surrounding joinable groups */
    doc_formatting&
    surround_joinable(const string& prefix, const string& postfix) {
        joinablePre_ = prefix;
        joinablePst_ = postfix;
        return *this;
    }
    const string& joinable_prefix()  const noexcept { return joinablePre_; }
    const string& joinable_postfix() const noexcept { return joinablePst_; }

    /** @brief determines maximum number of flags per parameter to be printed
     *         in detailed parameter documentation lines */
    doc_formatting& max_flags_per_param_in_doc(int max) {
        maxAltInDocs_ = max > 0 ? max : 0;
        return *this;
    }
    int max_flags_per_param_in_doc() const noexcept { return maxAltInDocs_; }

    /** @brief determines maximum number of flags per parameter to be printed
     *         in usage lines */
    doc_formatting& max_flags_per_param_in_usage(int max) {
        maxAltInUsage_ = max > 0 ? max : 0;
        return *this;
    }
    int max_flags_per_param_in_usage() const noexcept { return maxAltInUsage_; }

    /** @brief determines number of empty rows after one single-line
     *         documentation entry */
    doc_formatting& line_spacing(int lines) {
        lineSpc_ = lines > 0 ? lines : 0;
        return *this;
    }
    int line_spacing() const noexcept { return lineSpc_; }

    /** @brief determines number of empty rows before and after a paragraph;
     *         a paragraph is defined by a documented group or if
     *         a parameter documentation entry used more than one line */
    doc_formatting& paragraph_spacing(int lines) {
        paragraphSpc_ = lines > 0 ? lines : 0;
        return *this;
    }
    int paragraph_spacing() const noexcept { return paragraphSpc_; }

    /** @brief determines if alternative flags with a common prefix should
     *         be printed in a merged fashion */
    doc_formatting& merge_alternative_flags_with_common_prefix(bool yes = true) {
        mergeAltCommonPfx_ = yes;
        return *this;
    }
    bool merge_alternative_flags_with_common_prefix() const noexcept {
        return mergeAltCommonPfx_;
    }

    /** @brief determines if joinable flags with a common prefix should
     *         be printed in a merged fashion */
    doc_formatting& merge_joinable_with_common_prefix(bool yes = true) {
        mergeJoinableCommonPfx_ = yes;
        return *this;
    }
    bool merge_joinable_with_common_prefix() const noexcept {
        return mergeJoinableCommonPfx_;
    }

    /** @brief determines if children of exclusive groups should be printed
     *         on individual lines if the exceed 'alternatives_min_split_size'
     */
    doc_formatting& split_alternatives(bool yes = true) {
        splitTopAlt_ = yes;
        return *this;
    }
    bool split_alternatives() const noexcept {
        return splitTopAlt_;
    }

    /** @brief determines how many children exclusive groups can have before
     *         their children are printed on individual usage lines */
    doc_formatting& alternatives_min_split_size(int size) {
        groupSplitSize_ = size > 0 ? size : 0;
        return *this;
    }
    int alternatives_min_split_size() const noexcept { return groupSplitSize_; }

    /** @brief determines whether to ignore new line characters in docstrings
     */
    doc_formatting& ignore_newline_chars(bool yes = true) {
        ignoreNewlines_ = yes;
        return *this;
    }
    bool ignore_newline_chars() const noexcept {
        return ignoreNewlines_;
    }

private:
    string paramSep_      = string(" ");
    string groupSep_      = string(" ");
    string altParamSep_   = string("|");
    string altGroupSep_   = string(" | ");
    string flagSep_       = string(", ");
    string labelPre_      = string("<");
    string labelPst_      = string(">");
    string optionPre_     = string("[");
    string optionPst_     = string("]");
    string repeatPre_     = string("");
    string repeatPst_     = string("...");
    string groupPre_      = string("(");
    string groupPst_      = string(")");
    string alternPre_     = string("(");
    string alternPst_     = string(")");
    string alternFlagPre_ = string("");
    string alternFlagPst_ = string("");
    string joinablePre_   = string("(");
    string joinablePst_   = string(")");
    string emptyLabel_    = string("");
    int firstCol_ = 8;
    int docCol_ = 20;
    int lastCol_ = 100;
    int indentSize_ = 4;
    int maxAltInUsage_ = 1;
    int maxAltInDocs_ = 32;
    int lineSpc_ = 0;
    int paragraphSpc_ = 1;
    int groupSplitSize_ = 3;
    bool splitTopAlt_ = true;
    bool mergeAltCommonPfx_ = false;
    bool mergeJoinableCommonPfx_ = true;
    bool ignoreNewlines_ = false;
};



namespace detail {

/*************************************************************************//**
 *
 * @brief stream wrapper that applies formatting like line wrapping
 *        to stream data
 *
 *****************************************************************************/
template<class OStream = std::ostream, class StringT = doc_string>
class formatting_ostream
{
public:
    using string_type = StringT;
    using size_type   = typename string_type::size_type;
    using char_type   = typename string_type::value_type;

    formatting_ostream(OStream& os):
        os_(os),
        curCol_{0}, firstCol_{0}, lastCol_{100},
        hangingIndent_{0}, paragraphSpacing_{0}, paragraphSpacingThreshold_{2},
        curBlankLines_{0}, curParagraphLines_{1},
        totalNonBlankLines_{0},
        ignoreInputNls_{false}
    {}


    //---------------------------------------------------------------
    const OStream& base() const noexcept { return os_; }
          OStream& base()       noexcept { return os_; }

    bool good() const { return os_.good(); }


    //---------------------------------------------------------------
    /** @brief determines the leftmost border of the text body */
    formatting_ostream& first_column(int c) {
        firstCol_ = c < 0 ? 0 : c;
        return *this;
    }
    int first_column() const noexcept { return firstCol_; }

    /** @brief determines the rightmost border of the text body */
    formatting_ostream& last_column(int c) {
        lastCol_ = c < 0 ? 0 : c;
        return *this;
    }

    int last_column() const noexcept { return lastCol_; }

    int text_width() const noexcept {
        return lastCol_ - firstCol_;
    }

    /** @brief additional indentation for the 2nd, 3rd, ... line of
               a paragraph (sequence of soft-wrapped lines) */
    formatting_ostream& hanging_indent(int amount) {
        hangingIndent_ = amount;
        return *this;
    }
    int hanging_indent() const noexcept {
        return hangingIndent_;
    }

    /** @brief amount of blank lines between paragraphs */
    formatting_ostream& paragraph_spacing(int lines) {
        paragraphSpacing_ = lines;
        return *this;
    }
    int paragraph_spacing() const noexcept {
        return paragraphSpacing_;
    }

    /** @brief insert paragraph spacing
               if paragraph is at least 'lines' lines long */
    formatting_ostream& min_paragraph_lines_for_spacing(int lines) {
        paragraphSpacingThreshold_ = lines;
        return *this;
    }
    int min_paragraph_lines_for_spacing() const noexcept {
        return paragraphSpacingThreshold_;
    }

    /** @brief if set to true, newline characters will be ignored */
    formatting_ostream& ignore_newline_chars(bool yes) {
        ignoreInputNls_ = yes;
        return *this;
    }

    bool ignore_newline_chars() const noexcept {
        return ignoreInputNls_;
    }


    //---------------------------------------------------------------
    /* @brief insert 'n' spaces */
    void write_spaces(int n) {
        if(n < 1) return;
        os_ << string_type(size_type(n), ' ');
        curCol_ += n;
    }

    /* @brief go to new line, but continue current paragraph */
    void wrap_soft(int times = 1) {
        if(times < 1) return;
        if(times > 1) {
            os_ << string_type(size_type(times), '\n');
        } else {
            os_ << '\n';
        }
        curCol_ = 0;
        ++curParagraphLines_;
    }

    /* @brief go to new line, and start a new paragraph */
    void wrap_hard(int times = 1) {
        if(times < 1) return;

        if(paragraph_spacing() > 0 &&
           paragraph_lines() >= min_paragraph_lines_for_spacing())
        {
            times = paragraph_spacing() + 1;
        }

        if(times > 1) {
            os_ << string_type(size_type(times), '\n');
            curBlankLines_ += times - 1;
        } else {
            os_ << '\n';
        }
        if(at_begin_of_line()) {
            ++curBlankLines_;
        }
        curCol_ = 0;
        curParagraphLines_ = 1;
    }


    //---------------------------------------------------------------
    bool at_begin_of_line() const noexcept {
        return curCol_ <= current_line_begin();
    }
    int current_line_begin() const noexcept {
        return in_hanging_part_of_paragraph()
            ? firstCol_ + hangingIndent_
            : firstCol_;
    }

    int current_column() const noexcept {
        return curCol_;
    }

    int total_non_blank_lines() const noexcept {
        return totalNonBlankLines_;
    }
    int paragraph_lines() const noexcept {
        return curParagraphLines_;
    }
    int blank_lines_before_paragraph() const noexcept {
        return curBlankLines_;
    }


    //---------------------------------------------------------------
    template<class T>
    friend formatting_ostream&
    operator << (formatting_ostream& os, const T& x) {
        os.write(x);
        return os;
    }

    void flush() {
        os_.flush();
    }


private:
    bool in_hanging_part_of_paragraph() const noexcept {
        return hanging_indent() > 0 && paragraph_lines() > 1;
    }
    bool current_line_empty() const noexcept {
        return curCol_ < 1;
    }
    bool left_of_text_area() const noexcept {
        return curCol_ < current_line_begin();
    }
    bool right_of_text_area() const noexcept {
        return curCol_ > lastCol_;
    }
    int columns_left_in_line() const noexcept {
        return lastCol_ - std::max(current_line_begin(), curCol_);
    }

    void fix_indent() {
        if(left_of_text_area()) {
            const auto fst = current_line_begin();
            write_spaces(fst - curCol_);
            curCol_ = fst;
        }
    }

    template<class Iter>
    bool only_whitespace(Iter first, Iter last) const {
        return last == std::find_if_not(first, last,
                [](char_type c) { return std::isspace(c); });
    }

    /** @brief write any object */
    template<class T>
    void write(const T& x) {
        std::ostringstream ss;
        ss << x;
        write(std::move(ss).str());
    }

    /** @brief write a stringstream */
    void write(const std::ostringstream& s) {
        write(s.str());
    }

    /** @brief write a string */
    void write(const string_type& s) {
        write(s.begin(), s.end());
    }

    /** @brief partition output into lines */
    template<class Iter>
    void write(Iter first, Iter last)
    {
        if(first == last) return;
        if(*first == '\n') {
            if(!ignore_newline_chars()) wrap_hard();
            ++first;
            if(first == last) return;
        }
        auto i = std::find(first, last, '\n');
        if(i != last) {
            if(ignore_newline_chars()) ++i;
            if(i != last) {
                write_line(first, i);
                write(i, last);
            }
        }
        else {
            write_line(first, last);
        }
    }

    /** @brief handle line wrapping due to column constraints */
    template<class Iter>
    void write_line(Iter first, Iter last)
    {
        if(first == last) return;
        if(only_whitespace(first, last)) return;

        if(right_of_text_area()) wrap_soft();

        if(at_begin_of_line()) {
            //discard whitespace, it we start a new line
            first = std::find_if(first, last,
                        [](char_type c) { return !std::isspace(c); });
            if(first == last) return;
        }

        const auto n = int(std::distance(first,last));
        const auto m = columns_left_in_line();
        //if text to be printed is too long for one line -> wrap
        if(n > m) {
            //break before word, if break is mid-word
            auto breakat = first + m;
            while(breakat > first && !std::isspace(*breakat)) --breakat;
            //could not find whitespace before word -> try after the word
            if(!std::isspace(*breakat) && breakat == first) {
                breakat = std::find_if(first+m, last,
                          [](char_type c) { return std::isspace(c); });
            }
            if(breakat > first) {
                if(curCol_ < 1) ++totalNonBlankLines_;
                fix_indent();
                std::copy(first, breakat, std::ostream_iterator<char_type>(os_));
                curBlankLines_ = 0;
            }
            if(breakat < last) {
                wrap_soft();
                write_line(breakat, last);
            }
        }
        else {
            if(curCol_ < 1) ++totalNonBlankLines_;
            fix_indent();
            std::copy(first, last, std::ostream_iterator<char_type>(os_));
            curCol_ += n;
            curBlankLines_ = 0;
        }
    }

    /** @brief write a single character */
    void write(char_type c)
    {
        if(c == '\n') {
            if(!ignore_newline_chars()) wrap_hard();
        }
        else {
            if(at_begin_of_line()) ++totalNonBlankLines_;
            fix_indent();
            os_ << c;
            ++curCol_;
        }
    }

    OStream& os_;
    int curCol_;
    int firstCol_;
    int lastCol_;
    int hangingIndent_;
    int paragraphSpacing_;
    int paragraphSpacingThreshold_;
    int curBlankLines_;
    int curParagraphLines_;
    int totalNonBlankLines_;
    bool ignoreInputNls_;
};


}




/*************************************************************************//**
 *
 * @brief   generates usage lines
 *
 * @details lazily evaluated
 *
 *****************************************************************************/
class usage_lines
{
public:
    using string = doc_string;

    usage_lines(const group& cli, string prefix = "",
                const doc_formatting& fmt = doc_formatting{})
    :
        cli_(cli), fmt_(fmt), prefix_(std::move(prefix))
    {
        if(!prefix_.empty()) prefix_ += ' ';
    }

    usage_lines(const group& cli, const doc_formatting& fmt):
        usage_lines(cli, "", fmt)
    {}

    usage_lines& ommit_outermost_group_surrounders(bool yes) {
        ommitOutermostSurrounders_ = yes;
        return *this;
    }
    bool ommit_outermost_group_surrounders() const {
        return ommitOutermostSurrounders_;
    }

    template<class OStream>
    inline friend OStream& operator << (OStream& os, const usage_lines& p) {
        p.write(os);
        return os;
    }

    string str() const {
        std::ostringstream os; os << *this; return os.str();
    }


private:
    using stream_t = detail::formatting_ostream<>;
    const group& cli_;
    doc_formatting fmt_;
    string prefix_;
    bool ommitOutermostSurrounders_ = false;


    //-----------------------------------------------------
    struct context {
        group::depth_first_traverser pos;
        std::stack<string> separators;
        std::stack<string> postfixes;
        int level = 0;
        const group* outermost = nullptr;
        bool linestart = false;
        bool useOutermost = true;
        int line = 0;

        bool is_singleton() const noexcept {
            return linestart && pos.is_last_in_path();
        }
        bool is_alternative() const noexcept {
            return pos.parent().exclusive();
        }
    };


    /***************************************************************//**
     *
     * @brief writes usage text for command line parameters
     *
     *******************************************************************/
    template<class OStream>
    void write(OStream& os) const
    {
        detail::formatting_ostream<OStream> fos(os);
        fos.first_column(fmt_.first_column());
        fos.last_column(fmt_.last_column());

        auto hindent = int(prefix_.size());
        if(fos.first_column() + hindent >= int(0.4 * fos.text_width())) {
            hindent = fmt_.indent_size();
        }
        fos.hanging_indent(hindent);

        fos.paragraph_spacing(fmt_.paragraph_spacing());
        fos.min_paragraph_lines_for_spacing(2);
        fos.ignore_newline_chars(fmt_.ignore_newline_chars());

        context cur;
        cur.pos = cli_.begin_dfs();
        cur.linestart = true;
        cur.level = cur.pos.level();
        cur.outermost = &cli_;

        write(fos, cur, prefix_);
    }


    /***************************************************************//**
     *
     * @brief writes usage text for command line parameters
     *
     * @param prefix   all that goes in front of current things to print
     *
     *******************************************************************/
    template<class OStream>
    void write(OStream& os, context cur, string prefix) const
    {
        if(!cur.pos) return;

        std::ostringstream buf;
        if(cur.linestart) buf << prefix;
        const auto initPos = buf.tellp();

        cur.level = cur.pos.level();

        if(cur.useOutermost) {
            //we cannot start outside of the outermost group
            //so we have to treat it separately
            start_group(buf, cur.pos.parent(), cur);
            if(!cur.pos) {
                os << buf.str();
                return;
            }
        }
        else {
            //don't visit siblings of starter node
            cur.pos.skip_siblings();
        }
        check_end_group(buf, cur);

        do {
            if(buf.tellp() > initPos) cur.linestart = false;
            if(!cur.linestart && !cur.pos.is_first_in_group()) {
                buf << cur.separators.top();
            }
            if(cur.pos->is_group()) {
                start_group(buf, cur.pos->as_group(), cur);
                if(!cur.pos) {
                    os << buf.str();
                    return;
                }
            }
            else {
                buf << param_label(cur.pos->as_param(), cur);
                ++cur.pos;
            }
            check_end_group(buf, cur);
        } while(cur.pos);

        os << buf.str();
    }


    /***************************************************************//**
     *
     * @brief handles pattern group surrounders and separators
     *        and alternative splitting
     *
     *******************************************************************/
    void start_group(std::ostringstream& os,
                     const group& group, context& cur) const
    {
        //does cur.pos already point to a member or to group itself?
        //needed for special treatment of outermost group
        const bool alreadyInside = &(cur.pos.parent()) == &group;

        auto lbl = joined_label(group, cur);
        if(!lbl.empty()) {
            os << lbl;
            cur.linestart = false;
            //skip over entire group as its label has already been created
            if(alreadyInside) {
                cur.pos.next_after_siblings();
            } else {
                cur.pos.next_sibling();
            }
        }
        else {
            const bool splitAlternatives = group.exclusive() &&
                fmt_.split_alternatives() &&
                std::any_of(group.begin(), group.end(),
                    [this](const pattern& p) {
                        return int(p.param_count()) >= fmt_.alternatives_min_split_size();
                    });

            if(splitAlternatives) {
                cur.postfixes.push("");
                cur.separators.push("");
                //recursively print alternative paths in decision-DAG
                //enter group?
                if(!alreadyInside) ++cur.pos;
                cur.linestart = true;
                cur.useOutermost = false;
                auto pfx = os.str();
                os.str("");
                //print paths in DAG starting at each group member
                for(std::size_t i = 0; i < group.size(); ++i) {
                    std::stringstream buf;
                    cur.outermost = cur.pos->is_group() ? &(cur.pos->as_group()) : nullptr;
                    write(buf, cur, pfx);
                    if(buf.tellp() > int(pfx.size())) {
                        os << buf.str();
                        if(i < group.size()-1) {
                            if(cur.line > 0) {
                                os << string(fmt_.line_spacing(), '\n');
                            }
                            ++cur.line;
                            os << '\n';
                        }
                    }
                    cur.pos.next_sibling(); //do not descend into memebers
                }
                cur.pos.invalidate(); //signal end-of-path
                return;
            }
            else {
                //pre & postfixes, separators
                auto surround = group_surrounders(group, cur);
                os << surround.first;
                cur.postfixes.push(std::move(surround.second));
                cur.separators.push(group_separator(group, fmt_));
                //descend into group?
                if(!alreadyInside) ++cur.pos;
            }
        }
        cur.level = cur.pos.level();
    }


    /***************************************************************//**
     *
     *******************************************************************/
    void check_end_group(std::ostringstream& os, context& cur) const
    {
        for(; cur.level > cur.pos.level(); --cur.level) {
            os << cur.postfixes.top();
            cur.postfixes.pop();
            cur.separators.pop();
        }
        cur.level = cur.pos.level();
    }


    /***************************************************************//**
     *
     * @brief makes usage label for one command line parameter
     *
     *******************************************************************/
    string param_label(const parameter& p, const context& cur) const
    {
        const auto& parent = cur.pos.parent();

        const bool startsOptionalSequence =
            parent.size() > 1 && p.blocking() && cur.pos.is_first_in_group();

        const bool outermost =
            ommitOutermostSurrounders_ && cur.outermost == &parent;

        const bool showopt = !cur.is_alternative() && !p.required()
            && !startsOptionalSequence && !outermost;

        const bool showrep = p.repeatable() && !outermost;

        string lbl;

        if(showrep) lbl += fmt_.repeat_prefix();
        if(showopt) lbl += fmt_.optional_prefix();

        const auto& flags = p.flags();
        if(!flags.empty()) {
            const int n = std::min(fmt_.max_flags_per_param_in_usage(),
                                   int(flags.size()));

            const bool surrAlt = n > 1 && !showopt && !cur.is_singleton();

            if(surrAlt) lbl += fmt_.alternative_flags_prefix();
            bool sep = false;
            for(int i = 0; i < n; ++i) {
                if(sep) {
                    if(cur.is_singleton())
                        lbl += fmt_.alternative_group_separator();
                    else
                        lbl += fmt_.flag_separator();
                }
                lbl += flags[i];
                sep = true;
            }
            if(surrAlt) lbl += fmt_.alternative_flags_postfix();
        }
        else {
             if(!p.label().empty()) {
                 lbl += fmt_.label_prefix()
                     + p.label()
                     + fmt_.label_postfix();
             } else if(!fmt_.empty_label().empty()) {
                 lbl += fmt_.label_prefix()
                     + fmt_.empty_label()
                     + fmt_.label_postfix();
             } else {
                 return "";
             }
        }

        if(showopt) lbl += fmt_.optional_postfix();
        if(showrep) lbl += fmt_.repeat_postfix();

        return lbl;
    }


    /***************************************************************//**
     *
     * @brief prints flags in one group in a merged fashion
     *
     *******************************************************************/
    string joined_label(const group& g, const context& cur) const
    {
        if(!fmt_.merge_alternative_flags_with_common_prefix() &&
           !fmt_.merge_joinable_with_common_prefix()) return "";

        const bool flagsonly = std::all_of(g.begin(), g.end(),
            [](const pattern& p){
                return p.is_param() && !p.as_param().flags().empty();
            });

        if(!flagsonly) return "";

        const bool showOpt = g.all_optional() &&
            !(ommitOutermostSurrounders_ && cur.outermost == &g);

        auto pfx = g.common_flag_prefix();
        if(pfx.empty()) return "";

        const auto n = pfx.size();
        if(g.exclusive() &&
           fmt_.merge_alternative_flags_with_common_prefix())
        {
            string lbl;
            if(showOpt) lbl += fmt_.optional_prefix();
            lbl += pfx + fmt_.alternatives_prefix();
            bool first = true;
            for(const auto& p : g) {
                if(p.is_param()) {
                    if(first)
                        first = false;
                    else
                        lbl += fmt_.alternative_param_separator();
                    lbl += p.as_param().flags().front().substr(n);
                }
            }
            lbl += fmt_.alternatives_postfix();
            if(showOpt) lbl += fmt_.optional_postfix();
            return lbl;
        }
        //no alternatives, but joinable flags
        else if(g.joinable() &&
            fmt_.merge_joinable_with_common_prefix())
        {
            const bool allSingleChar = std::all_of(g.begin(), g.end(),
                [&](const pattern& p){
                    return p.is_param() &&
                        p.as_param().flags().front().substr(n).size() == 1;
                });

            if(allSingleChar) {
                string lbl;
                if(showOpt) lbl += fmt_.optional_prefix();
                lbl += pfx;
                for(const auto& p : g) {
                    if(p.is_param())
                        lbl += p.as_param().flags().front().substr(n);
                }
                if(showOpt) lbl += fmt_.optional_postfix();
                return lbl;
            }
        }

        return "";
    }


    /***************************************************************//**
     *
     * @return symbols with which to surround a group
     *
     *******************************************************************/
    std::pair<string,string>
    group_surrounders(const group& group, const context& cur) const
    {
        string prefix;
        string postfix;

        const bool isOutermost = &group == cur.outermost;
        if(isOutermost && ommitOutermostSurrounders_)
            return {string{}, string{}};

        if(group.exclusive()) {
            if(group.all_optional()) {
                prefix  = fmt_.optional_prefix();
                postfix = fmt_.optional_postfix();
                if(group.all_flagless()) {
                    prefix  += fmt_.label_prefix();
                    postfix = fmt_.label_prefix() + postfix;
                }
            } else if(group.all_flagless()) {
                prefix  = fmt_.label_prefix();
                postfix = fmt_.label_postfix();
            } else if(!cur.is_singleton() || !isOutermost) {
                prefix  = fmt_.alternatives_prefix();
                postfix = fmt_.alternatives_postfix();
            }
        }
        else if(group.size() > 1 &&
                group.front().blocking() && !group.front().required())
        {
            prefix  = fmt_.optional_prefix();
            postfix = fmt_.optional_postfix();
        }
        else if(group.size() > 1 && cur.is_alternative() &&
                &group != cur.outermost)
        {
            prefix  = fmt_.group_prefix();
            postfix = fmt_.group_postfix();
        }
        else if(!group.exclusive() &&
            group.joinable() && !cur.linestart)
        {
            prefix  = fmt_.joinable_prefix();
            postfix = fmt_.joinable_postfix();
        }

        if(group.repeatable()) {
            if(prefix.empty()) prefix = fmt_.group_prefix();
            prefix = fmt_.repeat_prefix() + prefix;
            if(postfix.empty()) postfix = fmt_.group_postfix();
            postfix += fmt_.repeat_postfix();
        }

        return {std::move(prefix), std::move(postfix)};
    }


    /***************************************************************//**
     *
     * @return symbol that separates members of a group
     *
     *******************************************************************/
    static string
    group_separator(const group& group, const doc_formatting& fmt)
    {
        const bool only1ParamPerMember = std::all_of(group.begin(), group.end(),
            [](const pattern& p) { return p.param_count() < 2; });

        if(only1ParamPerMember) {
            if(group.exclusive()) {
                return fmt.alternative_param_separator();
            } else {
                return fmt.param_separator();
            }
        }
        else { //there is at least one large group inside
            if(group.exclusive()) {
                return fmt.alternative_group_separator();
            } else {
                return fmt.group_separator();
            }
        }
    }
};




/*************************************************************************//**
 *
 * @brief   generates parameter and group documentation from docstrings
 *
 * @details lazily evaluated
 *
 *****************************************************************************/
class documentation
{
public:
    using string          = doc_string;
    using filter_function = std::function<bool(const parameter&)>;

    documentation(const group& cli,
                  const doc_formatting& fmt = doc_formatting{},
                  filter_function filter = param_filter{})
    :
        cli_(cli), fmt_{fmt}, usgFmt_{fmt}, filter_{std::move(filter)}
    {
        //necessary, because we re-use "usage_lines" to generate
        //labels for documented groups
        usgFmt_.max_flags_per_param_in_usage(
            usgFmt_.max_flags_per_param_in_doc());
    }

    documentation(const group& cli, filter_function filter) :
        documentation{cli, doc_formatting{}, std::move(filter)}
    {}

    documentation(const group& cli, const param_filter& filter) :
        documentation{cli, doc_formatting{},
                      [filter](const parameter& p) { return filter(p); }}
    {}

    template<class OStream>
    inline friend OStream& operator << (OStream& os, const documentation& p) {
        p.write(os);
        return os;
    }

    string str() const {
        std::ostringstream os;
        write(os);
        return os.str();
    }


private:
    using dfs_traverser = group::depth_first_traverser;

    const group& cli_;
    doc_formatting fmt_;
    doc_formatting usgFmt_;
    filter_function filter_;
    enum class paragraph { param, group };


    /***************************************************************//**
     *
     * @brief writes documentation to output stream
     *
     *******************************************************************/
     template<class OStream>
     void write(OStream& os) const {
        detail::formatting_ostream<OStream> fos(os);
        fos.first_column(fmt_.first_column());
        fos.last_column(fmt_.last_column());
        fos.hanging_indent(0);
        fos.paragraph_spacing(0);
        fos.ignore_newline_chars(fmt_.ignore_newline_chars());
        print_doc(fos, cli_);
     }


    /***************************************************************//**
     *
     * @brief writes full documentation text for command line parameters
     *
     *******************************************************************/
    template<class OStream>
    void print_doc(detail::formatting_ostream<OStream>& os,
                   const group& cli, int indentLvl = 0) const
    {
        if(cli.empty()) return;

        //if group itself doesn't have docstring
        if(cli.doc().empty()) {
            for(const auto& p : cli) {
                print_doc(os, p, indentLvl);
            }
        }
        else { //group itself does have docstring
            bool anyDocInside = std::any_of(cli.begin(), cli.end(),
                [](const pattern& p){ return !p.doc().empty(); });

            if(anyDocInside) { //group docstring as title, then child entries
                handle_spacing(os, paragraph::group, indentLvl);
                os << cli.doc();
                for(const auto& p : cli) {
                    print_doc(os, p, indentLvl + 1);
                }
            }
            else { //group label first then group docstring
                auto lbl = usage_lines(cli, usgFmt_)
                           .ommit_outermost_group_surrounders(true).str();

                str::trim(lbl);
                handle_spacing(os, paragraph::param, indentLvl);
                print_entry(os, lbl, cli.doc());
            }
        }
    }


    /***************************************************************//**
     *
     * @brief writes documentation text for one group or parameter
     *
     *******************************************************************/
    template<class OStream>
    void print_doc(detail::formatting_ostream<OStream>& os,
                   const pattern& ptrn, int indentLvl) const
    {
        if(ptrn.is_group()) {
            print_doc(os, ptrn.as_group(), indentLvl);
        }
        else {
            const auto& p = ptrn.as_param();
            if(!filter_(p)) return;

            handle_spacing(os, paragraph::param, indentLvl);
            print_entry(os, param_label(p, fmt_), p.doc());
        }
    }

    /***************************************************************//**
     *
     * @brief handles line and paragraph spacings
     *
     *******************************************************************/
    template<class OStream>
    void handle_spacing(detail::formatting_ostream<OStream>& os,
                        paragraph p, int indentLvl) const
    {
        const auto oldIndent = os.first_column();
        const auto indent = fmt_.first_column() + indentLvl * fmt_.indent_size();

        if(os.total_non_blank_lines() < 1) {
            os.first_column(indent);
            return;
        }

        if(os.paragraph_lines() > 1 || indent < oldIndent) {
            os.wrap_hard(fmt_.paragraph_spacing() + 1);
        } else {
            os.wrap_hard();
        }

        if(p == paragraph::group) {
            if(os.blank_lines_before_paragraph() < fmt_.paragraph_spacing()) {
                os.wrap_hard(fmt_.paragraph_spacing() - os.blank_lines_before_paragraph());
            }
        }
        else if(os.blank_lines_before_paragraph() < fmt_.line_spacing()) {
            os.wrap_hard(fmt_.line_spacing() - os.blank_lines_before_paragraph());
        }
        os.first_column(indent);
    }

    /*********************************************************************//**
     *
     * @brief prints one entry = label + docstring
     *
     ************************************************************************/
    template<class OStream>
    void print_entry(detail::formatting_ostream<OStream>& os,
                     const string& label, const string& docstr) const
    {
        if(label.empty()) return;

        os << label;

        if(!docstr.empty()) {
            if(os.current_column() >= fmt_.doc_column()) os.wrap_soft();
            const auto oldcol = os.first_column();
            os.first_column(fmt_.doc_column());
            os << docstr;
            os.first_column(oldcol);
        }
    }


    /*********************************************************************//**
     *
     * @brief makes label for one parameter
     *
     ************************************************************************/
    static doc_string
    param_label(const parameter& param, const doc_formatting& fmt)
    {
        doc_string lbl;

        if(param.repeatable()) lbl += fmt.repeat_prefix();

        const auto& flags = param.flags();
        if(!flags.empty()) {
            lbl += flags[0];
            const int n = std::min(fmt.max_flags_per_param_in_doc(),
                                   int(flags.size()));
            for(int i = 1; i < n; ++i) {
                lbl += fmt.flag_separator() + flags[i];
            }
        }
        else if(!param.label().empty() || !fmt.empty_label().empty()) {
            lbl += fmt.label_prefix();
            if(!param.label().empty()) {
                lbl += param.label();
            } else {
                lbl += fmt.empty_label();
            }
            lbl += fmt.label_postfix();
        }

        if(param.repeatable()) lbl += fmt.repeat_postfix();

        return lbl;
    }

};




/*************************************************************************//**
 *
 * @brief stores strings for man page sections
 *
 *****************************************************************************/
class man_page
{
public:
    //---------------------------------------------------------------
    using string = doc_string;

    //---------------------------------------------------------------
    /** @brief man page section */
    class section {
    public:
        using string = doc_string;

        section(string stitle, string scontent):
            title_{std::move(stitle)}, content_{std::move(scontent)}
        {}

        const string& title()   const noexcept { return title_; }
        const string& content() const noexcept { return content_; }

    private:
        string title_;
        string content_;
    };

private:
    using section_store = std::vector<section>;

public:
    //---------------------------------------------------------------
    using value_type     = section;
    using const_iterator = section_store::const_iterator;
    using size_type      = section_store::size_type;


    //---------------------------------------------------------------
    man_page&
    append_section(string title, string content)
    {
        sections_.emplace_back(std::move(title), std::move(content));
        return *this;
    }
    //-----------------------------------------------------
    man_page&
    prepend_section(string title, string content)
    {
        sections_.emplace(sections_.begin(),
                          std::move(title), std::move(content));
        return *this;
    }


    //---------------------------------------------------------------
    const section& operator [] (size_type index) const noexcept {
        return sections_[index];
    }

    //---------------------------------------------------------------
    size_type size() const noexcept { return sections_.size(); }

    bool empty() const noexcept { return sections_.empty(); }


    //---------------------------------------------------------------
    const_iterator begin() const noexcept { return sections_.begin(); }
    const_iterator end()   const noexcept { return sections_.end(); }


    //---------------------------------------------------------------
    man_page& program_name(const string& n) {
        progName_ = n;
        return *this;
    }
    man_page& program_name(string&& n) {
        progName_ = std::move(n);
        return *this;
    }
    const string& program_name() const noexcept {
        return progName_;
    }


    //---------------------------------------------------------------
    man_page& section_row_spacing(int rows) {
        sectionSpc_ = rows > 0 ? rows : 0;
        return *this;
    }
    int section_row_spacing() const noexcept { return sectionSpc_; }


private:
    int sectionSpc_ = 1;
    section_store sections_;
    string progName_;
};



/*************************************************************************//**
 *
 * @brief generates man sections from command line parameters
 *        with sections "synopsis" and "options"
 *
 *****************************************************************************/
inline man_page
make_man_page(const group& cli,
              doc_string progname = "",
              const doc_formatting& fmt = doc_formatting{})
{
    man_page man;
    man.append_section("SYNOPSIS", usage_lines(cli,progname,fmt).str());
    man.append_section("OPTIONS", documentation(cli,fmt).str());
    return man;
}



/*************************************************************************//**
 *
 * @brief   generates man page based on command line parameters
 *
 *****************************************************************************/
template<class OStream>
OStream&
operator << (OStream& os, const man_page& man)
{
    bool first = true;
    const auto secSpc = doc_string(man.section_row_spacing() + 1, '\n');
    for(const auto& section : man) {
        if(!section.content().empty()) {
            if(first) first = false; else os << secSpc;
            if(!section.title().empty()) os << section.title() << '\n';
            os << section.content();
        }
    }
    os << '\n';
    return os;
}





/*************************************************************************//**
 *
 * @brief printing methods for debugging command line interfaces
 *
 *****************************************************************************/
namespace debug {


/*************************************************************************//**
 *
 * @brief prints first flag or value label of a parameter
 *
 *****************************************************************************/
inline doc_string doc_label(const parameter& p)
{
    if(!p.flags().empty()) return p.flags().front();
    if(!p.label().empty()) return p.label();
    return doc_string{"<?>"};
}

inline doc_string doc_label(const group&)
{
    return "<group>";
}

inline doc_string doc_label(const pattern& p)
{
    return p.is_group() ? doc_label(p.as_group()) : doc_label(p.as_param());
}


/*************************************************************************//**
 *
 * @brief prints parsing result
 *
 *****************************************************************************/
template<class OStream>
void print(OStream& os, const parsing_result& result)
{
    for(const auto& m : result) {
        os << "#" << m.index() << " " << m.arg() << " -> ";
        auto p = m.param();
        if(p) {
            os << doc_label(*p) << " \t";
            if(m.repeat() > 0) {
                os << (m.bad_repeat() ? "[bad repeat " : "[repeat ")
                   <<  m.repeat() << "]";
            }
            if(m.blocked())  os << " [blocked]";
            if(m.conflict()) os << " [conflict]";
            os << '\n';
        }
        else {
            os << " [unmapped]\n";
        }
    }

    for(const auto& m : result.missing()) {
        auto p = m.param();
        if(p) {
            os << doc_label(*p) << " \t";
            os << " [missing after " << m.after_index() << "]\n";
        }
    }
}


/*************************************************************************//**
 *
 * @brief prints parameter label and some properties
 *
 *****************************************************************************/
template<class OStream>
void print(OStream& os, const parameter& p)
{
    if(p.blocking()) os << '!';
    if(!p.required()) os << '[';
    os << doc_label(p);
    if(p.repeatable()) os << "...";
    if(!p.required()) os << "]";
}


//-------------------------------------------------------------------
template<class OStream>
void print(OStream& os, const group& g, int level = 0);


/*************************************************************************//**
 *
 * @brief prints group or parameter; uses indentation
 *
 *****************************************************************************/
template<class OStream>
void print(OStream& os, const pattern& param, int level = 0)
{
    if(param.is_group()) {
        print(os, param.as_group(), level);
    }
    else {
        os << doc_string(4*level, ' ');
        print(os, param.as_param());
    }
}


/*************************************************************************//**
 *
 * @brief prints group and its contents; uses indentation
 *
 *****************************************************************************/
template<class OStream>
void print(OStream& os, const group& g, int level)
{
    auto indent = doc_string(4*level, ' ');
    os << indent;
    if(g.blocking()) os << '!';
    if(g.joinable()) os << 'J';
    os << (g.exclusive() ? "(|\n" : "(\n");
    for(const auto& p : g) {
        print(os, p, level+1);
    }
    os << '\n' << indent << (g.exclusive() ? "|)" : ")");
    if(g.repeatable()) os << "...";
    os << '\n';
}


} // namespace debug
} //namespace clipp

#endif

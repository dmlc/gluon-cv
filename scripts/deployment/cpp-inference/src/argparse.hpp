#ifndef ARGPARSE_HPP_
#define ARGPARSE_HPP_

#if __cplusplus >= 201103L
#include <unordered_map>
typedef std::unordered_map<std::string, size_t> IndexMap;
#else
#include <map>
typedef std::map<std::string, size_t> IndexMap;
#endif
#include <string>
#include <vector>
#include <typeinfo>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cassert>
#include <algorithm>

/*! @class ArgumentParser
 *  @brief A simple command-line argument parser based on the design of
 *  python's parser of the same name.
 *
 *  ArgumentParser is a simple C++ class that can parse arguments from
 *  the command-line or any array of strings. The syntax is familiar to
 *  anyone who has used python's ArgumentParser:
 *  \code
 *    // create a parser and add the options
 *    ArgumentParser parser;
 *    parser.addArgument("-n", "--name");
 *    parser.addArgument("--inputs", '+');
 *
 *    // parse the command-line arguments
 *    parser.parse(argc, argv);
 *
 *    // get the inputs and iterate over them
 *    string name = parser.retrieve("name");
 *    vector<string> inputs = parser.retrieve<vector<string>>("inputs");
 *  \endcode
 *
 */
class ArgumentParser {
private:
    class Any;
    class Argument;
    class PlaceHolder;
    class Holder;
    typedef std::string String;
    typedef std::vector<Any> AnyVector;
    typedef std::vector<String> StringVector;
    typedef std::vector<Argument> ArgumentVector;

    // --------------------------------------------------------------------------
    // Type-erasure internal storage
    // --------------------------------------------------------------------------
    class Any {
    public:
        // constructor
        Any() : content(0) {}
        // destructor
        ~Any() { delete content; }
        // INWARD CONVERSIONS
        Any(const Any& other) : content(other.content ? other.content->clone() : 0) {}
        template <typename ValueType>
        Any(const ValueType& other)
                : content(new Holder<ValueType>(other)) {}
        Any& swap(Any& other) {
            std::swap(content, other.content);
            return *this;
        }
        Any& operator=(const Any& rhs) {
            Any tmp(rhs);
            return swap(tmp);
        }
        template <typename ValueType>
        Any& operator=(const ValueType& rhs) {
            Any tmp(rhs);
            return swap(tmp);
        }
        // OUTWARD CONVERSIONS
        template <typename ValueType>
        ValueType* toPtr() const {
            return content->type_info() == typeid(ValueType)
                       ? &static_cast<Holder<ValueType>*>(content)->held_
                       : 0;
        }
        template <typename ValueType>
        ValueType& castTo() {
            if (!toPtr<ValueType>()) throw std::bad_cast();
            return *toPtr<ValueType>();
        }
        template <typename ValueType>
        const ValueType& castTo() const {
            if (!toPtr<ValueType>()) throw std::bad_cast();
            return *toPtr<ValueType>();
        }

    private:
        // Inner placeholder interface
        class PlaceHolder {
        public:
            virtual ~PlaceHolder() {}
            virtual const std::type_info& type_info() const = 0;
            virtual PlaceHolder* clone() const = 0;
        };
        // Inner template concrete instantiation of PlaceHolder
        template <typename ValueType>
        class Holder : public PlaceHolder {
        public:
            ValueType held_;
            Holder(const ValueType& value) : held_(value) {}
            virtual const std::type_info& type_info() const { return typeid(ValueType); }
            virtual PlaceHolder* clone() const { return new Holder(held_); }
        };
        PlaceHolder* content;
    };

    // --------------------------------------------------------------------------
    // Argument
    // --------------------------------------------------------------------------
    static String delimit(const String& name) {
        return String(std::min(name.size(), (size_t)2), '-').append(name);
    }
    static String strip(const String& name) {
        size_t begin = 0;
        begin += name.size() > 0 ? name[0] == '-' : 0;
        begin += name.size() > 3 ? name[1] == '-' : 0;
        return name.substr(begin);
    }
    static String upper(const String& in) {
        String out(in);
        std::transform(out.begin(), out.end(), out.begin(), ::toupper);
        return out;
    }
    static String escape(const String& in) {
        String out(in);
        if (in.find(' ') != std::string::npos) out = String("\"").append(out).append("\"");
        return out;
    }

    struct Argument {
        Argument() : short_name(""), name(""), optional(true), fixed_nargs(0), fixed(true) {}
        Argument(const String& _short_name, const String& _name, bool _optional, char nargs)
                : short_name(_short_name), name(_name), optional(_optional) {
            if (nargs == '+' || nargs == '*') {
                variable_nargs = nargs;
                fixed = false;
            } else {
                fixed_nargs = nargs;
                fixed = true;
            }
        }
        String short_name;
        String name;
        bool optional;
        union {
            size_t fixed_nargs;
            char variable_nargs;
        };
        bool fixed;
        String canonicalName() const { return (name.empty()) ? short_name : name; }
        String toString(bool named = true) const {
            std::ostringstream s;
            String uname = name.empty() ? upper(strip(short_name)) : upper(strip(name));
            if (named && optional) s << "[";
            if (named) s << canonicalName();
            if (fixed) {
                size_t N = std::min((size_t)3, fixed_nargs);
                for (size_t n = 0; n < N; ++n) s << " " << uname;
                if (N < fixed_nargs) s << " ...";
            }
            if (!fixed) {
                s << " ";
                if (variable_nargs == '*') s << "[";
                s << uname << " ";
                if (variable_nargs == '+') s << "[";
                s << uname << "...]";
            }
            if (named && optional) s << "]";
            return s.str();
        }
    };

    void insertArgument(const Argument& arg) {
        size_t N = arguments_.size();
        arguments_.push_back(arg);
        if (arg.fixed && arg.fixed_nargs <= 1) {
            variables_.push_back(String());
        } else {
            variables_.push_back(StringVector());
        }
        if (!arg.short_name.empty()) index_[arg.short_name] = N;
        if (!arg.name.empty()) index_[arg.name] = N;
        if (!arg.optional) required_++;
    }

    // --------------------------------------------------------------------------
    // Error handling
    // --------------------------------------------------------------------------
    void argumentError(const std::string& msg, bool show_usage = false) {
        if (use_exceptions_) throw std::invalid_argument(msg);
        std::cerr << "ArgumentParser error: " << msg << std::endl;
        if (show_usage) std::cerr << usage() << std::endl;
        exit(-5);
    }

    // --------------------------------------------------------------------------
    // Member variables
    // --------------------------------------------------------------------------
    IndexMap index_;
    bool ignore_first_;
    bool use_exceptions_;
    size_t required_;
    String app_name_;
    String final_name_;
    ArgumentVector arguments_;
    AnyVector variables_;

public:
    ArgumentParser() : ignore_first_(true), use_exceptions_(false), required_(0) {}
    // --------------------------------------------------------------------------
    // addArgument
    // --------------------------------------------------------------------------
    void appName(const String& name) { app_name_ = name; }
    void addArgument(const String& name, char nargs = 0, bool optional = true) {
        if (name.size() > 2) {
            Argument arg("", verify(name), optional, nargs);
            insertArgument(arg);
        } else {
            Argument arg(verify(name), "", optional, nargs);
            insertArgument(arg);
        }
    }
    void addArgument(const String& short_name, const String& name, char nargs = 0,
                     bool optional = true) {
        Argument arg(verify(short_name), verify(name), optional, nargs);
        insertArgument(arg);
    }
    void addFinalArgument(const String& name, char nargs = 1, bool optional = false) {
        final_name_ = delimit(name);
        Argument arg("", final_name_, optional, nargs);
        insertArgument(arg);
    }
    void ignoreFirstArgument(bool ignore_first) { ignore_first_ = ignore_first; }
    String verify(const String& name) {
        if (name.empty()) argumentError("argument names must be non-empty");
        if ((name.size() == 2 && name[0] != '-') || name.size() == 3)
            argumentError(String("invalid argument '")
                              .append(name)
                              .append("'. Short names must begin with '-'"));
        if (name.size() > 3 && (name[0] != '-' || name[1] != '-'))
            argumentError(String("invalid argument '")
                              .append(name)
                              .append("'. Multi-character names must begin with '--'"));
        return name;
    }

    // --------------------------------------------------------------------------
    // Parse
    // --------------------------------------------------------------------------
    void parse(size_t argc, const char** argv) { parse(StringVector(argv, argv + argc)); }

    void parse(const StringVector& argv) {
        // check if the app is named
        if (app_name_.empty() && ignore_first_ && !argv.empty()) app_name_ = argv[0];

        // set up the working set
        Argument active;
        Argument final = final_name_.empty() ? Argument() : arguments_[index_[final_name_]];
        size_t consumed = 0;
        size_t nrequired = final.optional ? required_ : required_ - 1;
        size_t nfinal = final.optional ? 0 : (final.fixed ? final.fixed_nargs
                                                          : (final.variable_nargs == '+' ? 1 : 0));

        // iterate over each element of the array
        for (StringVector::const_iterator in = argv.begin() + ignore_first_;
             in < argv.end() - nfinal; ++in) {
            String active_name = active.canonicalName();
            String el = *in;
            //  check if the element is a key
            if (index_.count(el) == 0) {
                // input
                // is the current active argument expecting more inputs?
                if (active.fixed && active.fixed_nargs <= consumed)
                    argumentError(String("attempt to pass too many inputs to ").append(active_name),
                                  true);
                if (active.fixed && active.fixed_nargs == 1) {
                    variables_[index_[active_name]].castTo<String>() = el;
                } else {
                    variables_[index_[active_name]].castTo<StringVector>().push_back(el);
                }
                consumed++;
            } else {
                // new key!
                // has the active argument consumed enough elements?
                if ((active.fixed && active.fixed_nargs != consumed) ||
                    (!active.fixed && active.variable_nargs == '+' && consumed < 1))
                    argumentError(String("encountered argument ")
                                      .append(el)
                                      .append(" when expecting more inputs to ")
                                      .append(active_name),
                                  true);
                active = arguments_[index_[el]];
                // check if we've satisfied the required arguments
                if (active.optional && nrequired > 0)
                    argumentError(String("encountered optional argument ")
                                      .append(el)
                                      .append(" when expecting more required arguments"),
                                  true);
                // are there enough arguments for the new argument to consume?
                if ((active.fixed && active.fixed_nargs > (argv.end() - in - nfinal - 1)) ||
                    (!active.fixed && active.variable_nargs == '+' &&
                     !(argv.end() - in - nfinal - 1)))
                    argumentError(String("too few inputs passed to argument ").append(el), true);
                if (!active.optional) nrequired--;
                consumed = 0;
            }
        }

        for (StringVector::const_iterator in =
                 std::max(argv.begin() + ignore_first_, argv.end() - nfinal);
             in != argv.end(); ++in) {
            String el = *in;
            // check if we accidentally find an argument specifier
            if (index_.count(el))
                argumentError(String("encountered argument specifier ")
                                  .append(el)
                                  .append(" while parsing final required inputs"),
                              true);
            if (final.fixed && final.fixed_nargs == 1) {
                variables_[index_[final_name_]].castTo<String>() = el;
            } else {
                variables_[index_[final_name_]].castTo<StringVector>().push_back(el);
            }
            nfinal--;
        }

        // check that all of the required arguments have been encountered
        if (nrequired > 0 || nfinal > 0)
            argumentError(String("too few required arguments passed to ").append(app_name_), true);
    }

    // --------------------------------------------------------------------------
    // Retrieve
    // --------------------------------------------------------------------------
    template <typename T>
    T& retrieve(const String& name) {
        if (index_.count(delimit(name)) == 0) throw std::out_of_range("Key not found");
        size_t N = index_[delimit(name)];
        return variables_[N].castTo<T>();
    }

    // --------------------------------------------------------------------------
    // Properties
    // --------------------------------------------------------------------------
    String usage() {
        // premable app name
        std::ostringstream help;
        help << "Usage: " << escape(app_name_);
        size_t indent = help.str().size();
        size_t linelength = 0;

        // get the required arguments
        for (ArgumentVector::const_iterator it = arguments_.begin(); it != arguments_.end(); ++it) {
            Argument arg = *it;
            if (arg.optional) continue;
            if (arg.name.compare(final_name_) == 0) continue;
            help << " ";
            String argstr = arg.toString();
            if (argstr.size() + linelength > 80) {
                help << "\n" << String(indent, ' ');
                linelength = 0;
            } else {
                linelength += argstr.size();
            }
            help << argstr;
        }

        // get the optional arguments
        for (ArgumentVector::const_iterator it = arguments_.begin(); it != arguments_.end(); ++it) {
            Argument arg = *it;
            if (!arg.optional) continue;
            if (arg.name.compare(final_name_) == 0) continue;
            help << " ";
            String argstr = arg.toString();
            if (argstr.size() + linelength > 80) {
                help << "\n" << String(indent, ' ');
                linelength = 0;
            } else {
                linelength += argstr.size();
            }
            help << argstr;
        }

        // get the final argument
        if (!final_name_.empty()) {
            Argument arg = arguments_[index_[final_name_]];
            String argstr = arg.toString(false);
            if (argstr.size() + linelength > 80) {
                help << "\n" << String(indent, ' ');
                linelength = 0;
            } else {
                linelength += argstr.size();
            }
            help << argstr;
        }

        return help.str();
    }
    void useExceptions(bool state) { use_exceptions_ = state; }
    bool empty() const { return index_.empty(); }
    void clear() {
        ignore_first_ = true;
        required_ = 0;
        index_.clear();
        arguments_.clear();
        variables_.clear();
    }
    bool exists(const String& name) const { return index_.count(delimit(name)) > 0; }
    size_t count(const String& name) {
        // check if the name is an argument
        if (index_.count(delimit(name)) == 0) return 0;
        size_t N = index_[delimit(name)];
        Argument arg = arguments_[N];
        Any var = variables_[N];
        // check if the argument is a vector
        if (arg.fixed) {
            return !var.castTo<String>().empty();
        } else {
            return var.castTo<StringVector>().size();
        }
    }
};
#endif
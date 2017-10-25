
namespace {
inline void print() { std::cout << std::endl; }

template <class T>
inline void print(const T &arg) {
  std::cout << arg << std::endl;
}

template <class T, typename... Args>
inline void print(T arg, Args... args) {
  std::cout << arg << " ";
  print(args...);
}
}

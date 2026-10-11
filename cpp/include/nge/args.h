// Tiny command-line parser: --key value / --flag
#pragma once
#include <cstdlib>
#include <string>
#include <vector>

namespace nge {

struct Args {
  std::vector<std::string> v;
  Args(int argc, char** argv) : v(argv + 1, argv + argc) {}
  bool has(const std::string& k) const { for (auto& a : v) if (a == k) return true; return false; }
  std::string get(const std::string& k, const std::string& def) const {
    for (size_t i = 0; i + 1 < v.size(); ++i) if (v[i] == k) return v[i + 1];
    return def;
  }
  int geti(const std::string& k, int def) const { return std::atoi(get(k, std::to_string(def)).c_str()); }
  double getd(const std::string& k, double def) const { return std::atof(get(k, std::to_string(def)).c_str()); }
};

}  // namespace nge

#pragma once
// =============================================================================
// config_loader.hpp — Thin yaml-cpp wrapper for tutorial configs
//
// Usage:
//   auto cfg = ConfigLoader::from_file("configs/tutorials/01_foo.yaml");
//   int n    = cfg.get<int>("tutorial", "N");
//   bool ok  = cfg.has("tutorial");
// =============================================================================

#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace cuda_tutorials {

class ConfigLoader {
public:
    // Factory — throws if the file cannot be read
    static ConfigLoader from_file(const std::string& path) {
        return ConfigLoader(YAML::LoadFile(path), path);
    }

    // Merge a second config file (e.g., global.yaml) into this one.
    // Keys already present in this config are NOT overwritten.
    void merge_defaults(const std::string& path) {
        YAML::Node defaults = YAML::LoadFile(path);
        merge_into(root_, defaults);
    }

    // -----------------------------------------------------------------------
    // Top-level key access: cfg.get<T>("key")
    // -----------------------------------------------------------------------
    template <typename T>
    T get(const std::string& key) const {
        if (!root_[key]) {
            throw std::runtime_error(
                "Config key not found: '" + key + "' in " + source_path_);
        }
        return root_[key].as<T>();
    }

    // -----------------------------------------------------------------------
    // Nested key access: cfg.get<T>("section", "key")
    // -----------------------------------------------------------------------
    template <typename T>
    T get(const std::string& section, const std::string& key) const {
        if (!root_[section]) {
            throw std::runtime_error(
                "Config section not found: '" + section + "' in " + source_path_);
        }
        if (!root_[section][key]) {
            throw std::runtime_error(
                "Config key not found: '" + section + "." + key +
                "' in " + source_path_);
        }
        return root_[section][key].as<T>();
    }

    // -----------------------------------------------------------------------
    // Existence check
    // -----------------------------------------------------------------------
    bool has(const std::string& key) const {
        return static_cast<bool>(root_[key]);
    }

    bool has(const std::string& section, const std::string& key) const {
        return root_[section] && static_cast<bool>(root_[section][key]);
    }

    // -----------------------------------------------------------------------
    // Vector helper — reads a YAML sequence as std::vector<T>
    // -----------------------------------------------------------------------
    template <typename T>
    std::vector<T> get_vector(const std::string& section,
                               const std::string& key) const {
        YAML::Node node;
        if (section.empty()) {
            if (!root_[key]) {
                throw std::runtime_error(
                    "Config key not found: '" + key + "' in " + source_path_);
            }
            node = root_[key];
        } else {
            if (!root_[section] || !root_[section][key]) {
                throw std::runtime_error(
                    "Config key not found: '" + section + "." + key +
                    "' in " + source_path_);
            }
            node = root_[section][key];
        }
        return node.as<std::vector<T>>();
    }

    // Convenience: get raw YAML node for a section (for advanced iteration)
    YAML::Node section(const std::string& key) const {
        return root_[key];
    }

private:
    explicit ConfigLoader(YAML::Node root, std::string path)
        : root_(std::move(root)), source_path_(std::move(path)) {}

    // Recursively copy keys from 'from' into 'to' without overwriting
    static void merge_into(YAML::Node& to, const YAML::Node& from) {
        if (!from.IsMap()) return;
        for (auto it = from.begin(); it != from.end(); ++it) {
            std::string key = it->first.as<std::string>();
            if (!to[key]) {
                to[key] = it->second;
            } else if (to[key].IsMap() && it->second.IsMap()) {
                YAML::Node sub = to[key];
                merge_into(sub, it->second);
            }
        }
    }

    YAML::Node  root_;
    std::string source_path_;
};

} // namespace cuda_tutorials

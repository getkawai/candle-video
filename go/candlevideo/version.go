package candlevideo

import "strings"

// SupportedBindingMajorVersion is the major ABI contract expected by this Go binding.
const SupportedBindingMajorVersion = "0"

// IsBindingVersionCompatible validates a semver-like version against the supported major version.
func IsBindingVersionCompatible(v string) bool {
	parts := strings.Split(v, ".")
	if len(parts) < 2 {
		return false
	}
	if parts[0] != SupportedBindingMajorVersion {
		return false
	}
	return parts[1] != ""
}

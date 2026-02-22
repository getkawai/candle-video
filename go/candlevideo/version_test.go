package candlevideo

import "testing"

func TestIsBindingVersionCompatible(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{in: "0.1.0", want: true},
		{in: "0.9.8", want: true},
		{in: "1.0.0", want: false},
		{in: "0", want: false},
		{in: "", want: false},
	}

	for _, tc := range cases {
		got := IsBindingVersionCompatible(tc.in)
		if got != tc.want {
			t.Fatalf("IsBindingVersionCompatible(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

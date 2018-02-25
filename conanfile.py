from conans import ConanFile


class FrugallyDeepConan(ConanFile):
    name = "frugally-deep"
    license = "MIT License"
    url = "https://github.com/Dobiasd/frugally-deep"
    description = "Header-only library for using Keras models in C++."
    exports_sources = ["include*", "LICENSE"]

    def package(self):
        self.copy("*LICENSE", dst="licenses")
        self.copy("*.h", src=".")
        self.copy("*.hpp", src=".")

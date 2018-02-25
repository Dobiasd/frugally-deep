from conans import ConanFile
import os

class FrugallyDeepConan(ConanFile):
    name = "frugally-deep"
    license = "MIT License"
    url = "https://github.com/Dobiasd/frugally-deep"
    description = "Header-only library for using Keras models in C++."
    exports_sources = ["include*", "LICENSE"]
    requires = ("eigen/3.3.4@conan/stable",
                "functionalplus/0.2@conan/stable",
                "jsonformoderncpp/3.1.0@vthiery/stable")

    def source(self):
        source_url = ("%s/archive/v%s.zip" % (self.url, self.version))
        tools.get(source_url)
        os.rename("%s-%s" % (self.name, self.version), "sources")

    def package(self):
        self.copy("*LICENSE", dst="licenses")
        self.copy("*.h", src=".")
        self.copy("*.hpp", src=".")

    def package_id(self):
        self.info.header_only()
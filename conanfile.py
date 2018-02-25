from conans import ConanFile, CMake, tools
import os


class FrugallyDeepConan(ConanFile):
    name = "frugally-deep"
    version = "0.1-p0"
    license = "The MIT License (MIT)"
    url = "https://github.com/Dobiasd/frugally-deep"
    description = "Use Keras models in C++ with ease"
    no_copy_source = True
    exports_sources = ["LICENSE"]
    requires = ("eigen/3.3.4@conan/stable",
                "functionalplus/0.2@conan/stable",
                "jsonformoderncpp/3.1.0@vthiery/stable")

    def source(self):
        source_url = ("%s/archive/v%s.zip" % (self.url, self.version))
        tools.get(source_url)
        os.rename("%s-%s" % (self.name, self.version), "sources")

    def package(self):
        self.copy("*LICENSE*", dst="licenses", src="sources")
        self.copy("*.h", dst=".", src="sources")
        self.copy("*.hpp", dst=".", src="sources")

    def package_id(self):
        self.info.header_only()

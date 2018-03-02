from conans import ConanFile


class FrugallyDeepConan(ConanFile):
    name = "frugally-deep"
    license = "The MIT License (MIT)"
    url = "https://github.com/Dobiasd/frugally-deep"
    description = "Use Keras models in C++ with ease"
    exports_sources = ["include*", "LICENSE"]
    requires = ("eigen/3.3.4@conan/stable",
                "functionalplus/v0.2-p1@dobiasd/stable",
                "jsonformoderncpp/3.1.0@vthiery/stable")

    def package(self):
        self.copy("*LICENSE*", dst="licenses")
        self.copy("*.h", dst=".")
        self.copy("*.hpp", dst=".")

## Chinese Translation of Stan Documentation

### Introduction

This repository contains the source files of the Chinese version of
[Stan documentation](https://mc-stan.org/docs/).
Right now we are working on the
[Stan User's Guide](https://mc-stan.org/docs/stan-users-guide/),
and you can build the translated book using the following command:

```bash
cd stan-users-guide
quarto render --to html
```

Alternatively, you can preview the rendered output
[here](https://stan-cn.netlify.app/).

The `upstream` directory contains the source files of the original
Stan documentation. We mainly use this directory to track the changes in
[the upstream repository](https://github.com/stan-dev/docs/tree/master/src).

### Acknowledgement

This project is supported by the
[NumFOCUS Small Development Grant](https://numfocus.org/programs/small-development-grants).

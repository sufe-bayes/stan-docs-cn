## Chinese Translation of Stan Documentation

This repository contains the source files of the Chinese version of
[Stan documentation](https://mc-stan.org/docs/).
Right now we are working on the
[Stan User's Guide](https://mc-stan.org/docs/stan-users-guide/),
and you can build the translated book using the following command:

```bash
cd stan-users-guide
quarto render --to html
```

The `upstream` directory contains the source files of the original
Stan documentation. We mainly use this directory to track the changes in
[the upstream repository](https://github.com/stan-dev/docs/tree/master/src).
